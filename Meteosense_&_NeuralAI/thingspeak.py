"""
thingspeak.py
=============
Conector universal para canales ThingSpeak.
Solo necesitas poner el ID del canal — auto-descubre campos y datos.

Uso:
    from thingspeak import ThingSpeakConnector
    ts = ThingSpeakConnector(3269359)
    df = ts.load()          # DataFrame con todos los datos
    info = ts.channel_info() # Metadatos del canal
"""

import time
import pandas as pd
import numpy as np
import requests
from typing import Optional, Dict, List
from config import THINGSPEAK_BASE_URL, FIELD_KEYWORDS, VALID_RANGES


class ThingSpeakConnector:
    """
    Conector inteligente para canales ThingSpeak.
    Auto-descubre los campos del canal y los mapea a variables conocidas.
    Soporta canales públicos y privados (con API key).
    """

    def __init__(self, channel_id: int, api_key: Optional[str] = None):
        self.channel_id = channel_id
        self.api_key = api_key
        self.base_url = THINGSPEAK_BASE_URL
        self._channel_meta: Optional[dict] = None
        self._field_map: Dict[str, str] = {}

    # ── Info del canal ───────────────────────────────────────────
    def channel_info(self) -> dict:
        """Obtiene metadatos del canal (nombre, campos, última actualización)."""
        if self._channel_meta:
            return self._channel_meta

        url = f"{self.base_url}/channels/{self.channel_id}/feeds.json"
        params = {"results": 0}
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            resp = self._request_with_retry(url, params)
            data = resp.json()
        except Exception as e:
            return {"error": str(e), "channel_id": self.channel_id}

        ch = data.get("channel", {})
        self._channel_meta = {
            "id":          ch.get("id", self.channel_id),
            "nombre":      ch.get("name", f"Canal {self.channel_id}"),
            "descripcion": ch.get("description", ""),
            "latitud":     ch.get("latitude", ""),
            "longitud":    ch.get("longitude", ""),
            "creado":      ch.get("created_at", ""),
            "actualizado": ch.get("updated_at", ""),
            "ultimo_entry": ch.get("last_entry_id", 0),
            "campos":      {},
        }

        # Descubrir campos field1..field8
        mapped_names_seen = set()
        for i in range(1, 9):
            field_name = ch.get(f"field{i}")
            if field_name:
                mapped = self._map_field_name(field_name)
                
                # Evitar nombres duplicados (ej: dos sensores de UV)
                original_mapped = mapped
                counter = 1
                while mapped in mapped_names_seen:
                    mapped = f"{original_mapped}_{counter}"
                    counter += 1
                mapped_names_seen.add(mapped)
                
                self._channel_meta["campos"][f"field{i}"] = {
                    "original": field_name,
                    "mapeado":  mapped,
                }
                self._field_map[f"field{i}"] = mapped

        return self._channel_meta

    # ── Carga de datos ───────────────────────────────────────────
    def load(self, results: int = 8000, days: Optional[int] = None) -> pd.DataFrame:
        """
        Descarga datos del canal y retorna un DataFrame con columnas mapeadas.
        
        Args:
            results: Número máximo de registros (ThingSpeak max = 8000)
            days: Si se especifica, descarga solo los últimos N días
        """
        # Asegurar que tenemos info del canal para mapear campos
        if not self._field_map:
            self.channel_info()

        url = f"{self.base_url}/channels/{self.channel_id}/feeds.json"
        params = {"results": min(results, 8000)}
        if self.api_key:
            params["api_key"] = self.api_key
        if days:
            params["days"] = days

        try:
            resp = self._request_with_retry(url, params)
            data = resp.json()
        except Exception as e:
            raise ConnectionError(
                f"No se pudo conectar al canal {self.channel_id}: {e}\n"
                f"Verifica tu conexión a internet y que el canal sea público."
            )

        feeds = data.get("feeds", [])
        if not feeds:
            raise ValueError(
                f"El canal {self.channel_id} no retornó datos. "
                f"Puede estar vacío o ser privado (usa api_key)."
            )

        df = pd.DataFrame(feeds)

        # Renombrar columnas field1..field8 a nombres descriptivos
        rename_map = {"created_at": "fecha"}
        for field_key, mapped_name in self._field_map.items():
            if field_key in df.columns:
                rename_map[field_key] = mapped_name
        df = df.rename(columns=rename_map)

        # Eliminar columna entry_id si existe
        df = df.drop(columns=["entry_id"], errors="ignore")

        # Convertir fecha
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

        # Convertir campos numéricos
        for col in df.columns:
            if col != "fecha":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Validar rangos meteorológicos
        df = self._validate_ranges(df)

        # Eliminar columnas que quedaron sin mapear (fieldN sin nombre)
        drop_cols = [c for c in df.columns if c.startswith("field")]
        df = df.drop(columns=drop_cols, errors="ignore")

        return df

    # ── Último valor ─────────────────────────────────────────────
    def last_value(self) -> dict:
        """Obtiene el último registro del canal."""
        url = f"{self.base_url}/channels/{self.channel_id}/feeds/last.json"
        params = {}
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            resp = self._request_with_retry(url, params)
            data = resp.json()
        except Exception:
            return {}

        result = {"fecha": data.get("created_at", "")}
        for field_key, mapped_name in self._field_map.items():
            val = data.get(field_key)
            if val is not None:
                try:
                    result[mapped_name] = float(val)
                except (ValueError, TypeError):
                    result[mapped_name] = val
        return result

    # ── Métodos internos ─────────────────────────────────────────
    def _map_field_name(self, original: str) -> str:
        """Mapea un nombre de campo de ThingSpeak a una variable estándar."""
        name_lower = original.strip().lower().replace(" ", "_").replace(".", "")
        
        for standard_name, keywords in FIELD_KEYWORDS.items():
            for kw in keywords:
                if kw in name_lower:
                    return standard_name
        
        # Si no se reconoce, usar el nombre normalizado
        return name_lower

    def _validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Marca como NaN los valores fuera de rangos físicamente válidos."""
        for col in df.columns:
            if col in VALID_RANGES:
                vmin, vmax = VALID_RANGES[col]
                mask = (df[col] < vmin) | (df[col] > vmax)
                if mask.any():
                    df.loc[mask, col] = np.nan
        return df

    def _request_with_retry(self, url: str, params: dict, retries: int = 3) -> requests.Response:
        """Hace request HTTP con reintentos y backoff exponencial."""
        last_error = None
        for attempt in range(retries):
            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                last_error = e
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s
        raise last_error

    def __repr__(self) -> str:
        info = self._channel_meta or {}
        name = info.get("nombre", f"Canal {self.channel_id}")
        n_campos = len(self._field_map)
        return f"ThingSpeakConnector({self.channel_id}: '{name}', {n_campos} campos)"

"""
station_manager.py
==================
Gestor multi-estación meteorológica con soporte ThingSpeak.
Conecta 7 estaciones: 1 canal real + 6 canales de ejemplo.
Solo necesitas poner el ID del canal para agregar una estación.

Uso:
    from station_manager import StationManager
    sm = StationManager()
    sm.load_all()                    # Carga todas las estaciones
    sm.station_summary()             # Resumen de todas
    sm.get_data(1)                   # Datos de estación 1
    sm.alerts()                      # Alertas meteorológicas activas
    sm.add_channel(12345)            # Agregar nueva estación
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from rich import box

from thingspeak import ThingSpeakConnector
from config import EXAMPLE_CHANNELS, ALERT_THRESHOLDS

console = Console()


class Station:
    """Representa una estación meteorológica individual."""

    def __init__(self, station_id: int, channel_id: int, nombre: str = "",
                 ubicacion: str = "", api_key: Optional[str] = None):
        self.station_id = station_id
        self.channel_id = channel_id
        self.nombre = nombre
        self.ubicacion = ubicacion
        self.connector = ThingSpeakConnector(channel_id, api_key=api_key)
        self.df: Optional[pd.DataFrame] = None
        self.last_update: Optional[datetime] = None
        self.status: str = "desconectada"
        self.error_msg: str = ""

    def load(self, results: int = 8000) -> Optional[pd.DataFrame]:
        """Carga datos de la estación desde ThingSpeak."""
        try:
            info = self.connector.channel_info()
            if "error" in info:
                self.status = "error"
                self.error_msg = info["error"]
                return None

            if not self.nombre:
                self.nombre = info.get("nombre", f"Estación {self.station_id}")

            self.df = self.connector.load(results=results)
            self.last_update = datetime.now()
            self.status = "conectada"
            self.error_msg = ""
            return self.df

        except Exception as e:
            self.status = "error"
            self.error_msg = str(e)
            return None

    def last_value(self) -> dict:
        """Obtiene el último valor de la estación."""
        try:
            return self.connector.last_value()
        except Exception:
            if self.df is not None and len(self.df) > 0:
                last = self.df.iloc[-1]
                return {col: last[col] for col in self.df.columns if col != "fecha"}
            return {}

    def get_fields(self) -> list:
        """Retorna los campos disponibles de esta estación."""
        info = self.connector.channel_info()
        campos = info.get("campos", {})
        return [v["mapeado"] for v in campos.values()]

    def __repr__(self) -> str:
        return (f"Station({self.station_id}: '{self.nombre}' "
                f"[{self.status}] canal={self.channel_id})")


class StationManager:
    """
    Gestor de múltiples estaciones meteorológicas.
    Administra conexiones ThingSpeak, carga datos y genera alertas.
    """

    def __init__(self):
        self.stations: Dict[int, Station] = {}
        self._init_default_stations()

    def _init_default_stations(self):
        """Inicializa las 7 estaciones por defecto desde config."""
        for sid, info in EXAMPLE_CHANNELS.items():
            self.stations[sid] = Station(
                station_id=sid,
                channel_id=info["id"],
                nombre=info["nombre"],
                ubicacion=info["ubicacion"],
            )

    # ── Carga de datos ───────────────────────────────────────────
    def load_all(self, results: int = 8000) -> Dict[int, str]:
        """
        Carga datos de todas las estaciones.
        Retorna dict {station_id: status}.
        """
        report = {}
        for sid, station in self.stations.items():
            console.print(f"  [dim]Cargando estación {sid}: {station.nombre}...[/dim]")
            df = station.load(results=results)
            if df is not None:
                report[sid] = f"✓ {len(df)} registros"
            else:
                report[sid] = f"✗ {station.error_msg[:60]}"
        return report

    def load_station(self, station_id: int, results: int = 8000) -> Optional[pd.DataFrame]:
        """Carga una estación específica."""
        if station_id not in self.stations:
            console.print(f"[red]Estación {station_id} no existe. Usa 'stations' para ver la lista.[/red]")
            return None
        return self.stations[station_id].load(results=results)

    # ── Datos ────────────────────────────────────────────────────
    def get_data(self, station_id: int) -> Optional[pd.DataFrame]:
        """Retorna el DataFrame de una estación."""
        station = self.stations.get(station_id)
        if station and station.df is not None:
            return station.df
        return None

    def get_all_data(self) -> Dict[int, pd.DataFrame]:
        """Retorna datos de todas las estaciones conectadas."""
        return {
            sid: st.df
            for sid, st in self.stations.items()
            if st.df is not None
        }

    # ── Agregar estaciones ───────────────────────────────────────
    def add_channel(self, channel_id: int, nombre: str = "",
                    ubicacion: str = "", api_key: Optional[str] = None) -> int:
        """
        Agrega una nueva estación por ID de canal ThingSpeak.
        Solo necesitas el ID — el sistema auto-descubre todo.
        Retorna el ID de estación asignado.
        """
        new_id = max(self.stations.keys(), default=0) + 1
        self.stations[new_id] = Station(
            station_id=new_id,
            channel_id=channel_id,
            nombre=nombre or f"Canal {channel_id}",
            ubicacion=ubicacion,
            api_key=api_key,
        )
        return new_id

    def remove_station(self, station_id: int):
        """Elimina una estación."""
        if station_id in self.stations:
            del self.stations[station_id]

    # ── Resumen ──────────────────────────────────────────────────
    def station_summary(self) -> List[dict]:
        """Resumen de todas las estaciones."""
        summary = []
        for sid, st in self.stations.items():
            info = {
                "id":          sid,
                "canal":       st.channel_id,
                "nombre":      st.nombre,
                "ubicacion":   st.ubicacion,
                "estado":      st.status,
                "registros":   len(st.df) if st.df is not None else 0,
                "campos":      ", ".join(st.get_fields()) if st.status == "conectada" else "—",
                "actualizado": st.last_update.strftime("%H:%M:%S") if st.last_update else "—",
            }
            summary.append(info)
        return summary

    # ── Alertas ──────────────────────────────────────────────────
    def alerts(self) -> List[dict]:
        """
        Revisa los últimos valores de todas las estaciones
        y genera alertas cuando se superan umbrales.
        """
        alertas = []
        for sid, station in self.stations.items():
            if station.status != "conectada" or station.df is None:
                continue

            last = station.last_value()
            for variable, thresholds in ALERT_THRESHOLDS.items():
                if variable in last:
                    value = last[variable]
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        label = thresholds["label"]
                        if value > thresholds["max"]:
                            alertas.append({
                                "estacion":  sid,
                                "nombre":    station.nombre,
                                "variable":  variable,
                                "valor":     round(value, 2),
                                "umbral":    f"> {thresholds['max']} {label}",
                                "nivel":     "Alto",
                                "mensaje":   f"{variable.capitalize()} excesiva: "
                                             f"{value:.1f} {label} (máx: {thresholds['max']})",
                            })
                        elif value < thresholds["min"]:
                            alertas.append({
                                "estacion":  sid,
                                "nombre":    station.nombre,
                                "variable":  variable,
                                "valor":     round(value, 2),
                                "umbral":    f"< {thresholds['min']} {label}",
                                "nivel":     "Alto",
                                "mensaje":   f"{variable.capitalize()} muy baja: "
                                             f"{value:.1f} {label} (mín: {thresholds['min']})",
                            })
        return alertas

    # ── Comparativa ──────────────────────────────────────────────
    def compare_stations(self, variable: str) -> Optional[pd.DataFrame]:
        """
        Crea tabla comparativa de una variable entre todas las estaciones conectadas.
        """
        rows = []
        for sid, station in self.stations.items():
            if station.df is None or variable not in station.df.columns:
                continue
            s = station.df[variable].dropna()
            if len(s) == 0:
                continue
            rows.append({
                "estacion":   sid,
                "nombre":     station.nombre,
                "min":        round(float(s.min()), 2),
                "max":        round(float(s.max()), 2),
                "media":      round(float(s.mean()), 2),
                "std":        round(float(s.std()), 2),
                "ultimo":     round(float(s.iloc[-1]), 2),
                "registros":  len(s),
            })
        return pd.DataFrame(rows) if rows else None

    def __repr__(self) -> str:
        n_total = len(self.stations)
        n_conn = sum(1 for s in self.stations.values() if s.status == "conectada")
        return f"StationManager({n_conn}/{n_total} estaciones conectadas)"

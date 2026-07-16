"""
processor.py
============
Análisis, limpieza e ingeniería de features del DataFrame cargado.
Motor de consultas en lenguaje natural simplificado.
Incluye detección automática de fechas, rolling features y validación pre-entrenamiento.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from config import MIN_ROWS_FOR_TRAINING


# ─────────────────────────────────────────────────────────────────
class DataProcessor:
    """Preprocesa y analiza el DataFrame para entrenar la red neuronal."""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.feature_names: list = []
        self.target_name: str = ""
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.label_encoders: dict = {}

    # ── Carga ─────────────────────────────────────────────────────
    def set_data(self, df: pd.DataFrame):
        self.df = df.copy()

    # ── Estadísticas ──────────────────────────────────────────────
    def summary(self) -> dict:
        if self.df is None:
            return {}
        num = self.df.select_dtypes(include=np.number)
        cat = self.df.select_dtypes(exclude=np.number)
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "numeric_cols": list(num.columns),
            "categorical_cols": list(cat.columns),
            "nulls": self.df.isnull().sum().to_dict(),
            "describe": num.describe().round(2).to_dict() if len(num.columns) else {},
        }

    def column_stats(self, col: str) -> dict:
        col = self._resolve_col(col)
        s = self.df[col]
        base = {"count": len(s), "nulls": s.isnull().sum(), "dtype": str(s.dtype)}
        if pd.api.types.is_numeric_dtype(s):
            base.update({
                "mean": round(s.mean(), 4),
                "median": round(s.median(), 4),
                "std": round(s.std(), 4),
                "min": round(s.min(), 4),
                "max": round(s.max(), 4),
                "q25": round(s.quantile(.25), 4),
                "q75": round(s.quantile(.75), 4),
            })
        else:
            base.update({
                "unique": s.nunique(),
                "top5": s.value_counts().head(5).to_dict(),
            })
        return base

    # ── Motor de consultas ────────────────────────────────────────
    def query(self, expression: str) -> pd.DataFrame:
        """
        Consultas en estilo pandas query o palabras clave simples:
          - "ventas > 50000"
          - "top 10 ventas"
          - "group by region sum ventas"
          - "sort by fecha"
          - "filter region == Norte"
        Matching de columnas case-insensitive.
        """
        expr = expression.strip().lower()
        df = self.df.copy()

        # top N col
        if expr.startswith("top "):
            parts = expr.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
            col = self._resolve_col(parts[2]) if len(parts) > 2 else self._first_numeric()
            if col and col in df.columns:
                return df.nlargest(n, col)
            return df.head(n)

        # group by col [agg] col2
        if "group by" in expr:
            tokens = expr.replace("group by", "groupby").split()
            idx = tokens.index("groupby")
            group_col = self._resolve_col(tokens[idx + 1]) if idx + 1 < len(tokens) else None
            agg = "sum"
            for a in ["sum", "mean", "count", "max", "min"]:
                if a in tokens:
                    agg = a
                    break
            val_col = None
            for t in tokens[idx + 2:]:
                if t not in ["sum", "mean", "count", "max", "min"]:
                    resolved = self._resolve_col(t)
                    if resolved:
                        val_col = resolved
                        break
            if group_col and group_col in df.columns:
                if val_col and val_col in df.columns:
                    return df.groupby(group_col)[val_col].agg(agg).reset_index()
                else:
                    return df.groupby(group_col).agg(agg, numeric_only=True).reset_index()

        # sort by col [desc]
        if expr.startswith("sort"):
            parts = expr.split()
            col = None
            for p in reversed(parts):
                if p not in ("sort", "by", "desc", "asc"):
                    col = self._resolve_col(p)
                    if col:
                        break
            if not col:
                col = self._first_numeric()
            asc = "desc" not in parts
            if col and col in df.columns:
                return df.sort_values(col, ascending=asc)
            return df

        # filter col op value  /  pandas expression
        try:
            return df.query(expression)
        except Exception:
            pass

        # Fallback: buscar coincidencia de columna
        for col in df.columns:
            if col in expr:
                return df[[col]].describe().T

        return df.head(20)

    # ── Preprocesado para red neuronal ────────────────────────────
    def preprocess(
        self,
        target_col: str,
        drop_cols: Optional[list] = None,
        add_lag_features: bool = True,
        add_rolling_features: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesa para entrenamiento:
        - Detecta y extrae features de columnas datetime
        - Codifica categóricas (LabelEncoder o frequency encoding)
        - Agrega lag features y rolling features
        - Escala X e y
        - Valida datos antes de entrenar
        Retorna (X, y) en escala [0,1].
        """
        target_col = self._resolve_col(target_col)
        if target_col not in self.df.columns:
            raise KeyError(f"Columna target '{target_col}' no existe. "
                           f"Disponibles: {list(self.df.columns)}")

        drop_cols = drop_cols or []
        df = self.df.drop(columns=drop_cols, errors="ignore").copy()
        df = df.dropna(subset=[target_col])
        self.target_name = target_col

        # ── Detectar y procesar columnas datetime ────────────────
        for col in df.columns:
            if col == target_col:
                continue
            if df[col].dtype == "object":
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                    if parsed.notna().sum() > len(df) * 0.5:
                        df[f"{col}_anio"] = parsed.dt.year
                        df[f"{col}_mes"] = parsed.dt.month
                        df[f"{col}_dia_semana"] = parsed.dt.dayofweek
                        df[f"{col}_trimestre"] = parsed.dt.quarter
                        df[f"{col}_fin_semana"] = (parsed.dt.dayofweek >= 5).astype(int)
                        df = df.drop(columns=[col])
                except Exception:
                    pass
            elif hasattr(df[col].dtype, "kind") and df[col].dtype.kind == "M":
                # Ya es datetime
                df[f"{col}_anio"] = df[col].dt.year
                df[f"{col}_mes"] = df[col].dt.month
                df[f"{col}_dia_semana"] = df[col].dt.dayofweek
                df[f"{col}_trimestre"] = df[col].dt.quarter
                df[f"{col}_fin_semana"] = (df[col].dt.dayofweek >= 5).astype(int)
                df = df.drop(columns=[col])

        # ── Lag features sobre el target ─────────────────────────
        if add_lag_features and pd.api.types.is_numeric_dtype(df[target_col]):
            for lag in [1, 2, 3]:
                df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)

        # ── Rolling features ─────────────────────────────────────
        if add_rolling_features and pd.api.types.is_numeric_dtype(df[target_col]):
            df[f"{target_col}_media_movil3"] = df[target_col].rolling(3).mean()
            df[f"{target_col}_std_movil3"] = df[target_col].rolling(3).std()

        df = df.dropna()

        # ── Validación pre-entrenamiento ─────────────────────────
        if len(df) < MIN_ROWS_FOR_TRAINING:
            raise ValueError(
                f"Dataset muy pequeño: {len(df)} filas (mínimo {MIN_ROWS_FOR_TRAINING}). "
                f"Agrega más datos o reduce los lag features."
            )
        if df[target_col].std() == 0:
            raise ValueError(
                f"La columna target '{target_col}' es constante (std=0). "
                f"No se puede entrenar un modelo con un target que no varía."
            )

        y_raw = df[target_col].values.reshape(-1, 1).astype(np.float32)
        X_df = df.drop(columns=[target_col])

        # ── Eliminar columnas 100% nulas ─────────────────────────
        null_cols = [c for c in X_df.columns if X_df[c].isna().all()]
        if null_cols:
            X_df = X_df.drop(columns=null_cols)

        # ── Codificar categóricas ────────────────────────────────
        for col in X_df.select_dtypes(include=["object", "category"]).columns:
            n_unique = X_df[col].nunique()
            if n_unique > 50:
                # Frequency encoding para alta cardinalidad
                freq = X_df[col].value_counts(normalize=True)
                X_df[col] = X_df[col].map(freq).fillna(0)
            else:
                le = LabelEncoder()
                X_df[col] = le.fit_transform(X_df[col].astype(str))
                self.label_encoders[col] = le

        # ── Rellenar NaN restantes ───────────────────────────────
        X_df = X_df.ffill()  # Forward fill primero
        X_df = X_df.fillna(X_df.median(numeric_only=True))  # Luego mediana
        X_df = X_df.fillna(0)  # Último recurso

        self.feature_names = list(X_df.columns)

        X = self.scaler_X.fit_transform(X_df.values.astype(np.float32))
        y = self.scaler_y.fit_transform(y_raw).flatten()

        return X, y

    def inverse_target(self, y_scaled: np.ndarray) -> np.ndarray:
        return self.scaler_y.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).flatten()

    # ── Helpers ───────────────────────────────────────────────────
    def _first_numeric(self) -> Optional[str]:
        """Retorna la primera columna numérica o None."""
        if self.df is None:
            return None
        nums = self.df.select_dtypes(include=np.number).columns
        return nums[0] if len(nums) > 0 else None

    def _resolve_col(self, name: str) -> Optional[str]:
        """Resuelve un nombre de columna case-insensitive."""
        if self.df is None:
            return name
        name_lower = name.strip().lower()
        for col in self.df.columns:
            if col.lower() == name_lower:
                return col
        return name  # Retorna original si no se encuentra

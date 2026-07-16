"""
predictor.py
============
Motor de predicciones: entrena la red y genera pronósticos futuros.
Incluye intervalos de confianza bootstrap, split temporal y protección de scalers.
"""

import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Optional, List

from neural_network import NeuralNet, Trainer
from processor import DataProcessor
from config import (
    DEFAULT_LAYERS, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE,
    CONFIDENCE_BOOTSTRAP_N, CONFIDENCE_NOISE_STD,
)


class Predictor:
    def __init__(self, processor: DataProcessor):
        self.proc = processor
        self.trainer: Optional[Trainer] = None
        self.metrics: dict = {}
        self.target_col: str = ""
        # Datos cacheados para evitar re-fitear scalers
        self._X_cached: Optional[np.ndarray] = None
        self._y_cached: Optional[np.ndarray] = None

    # ── Entrenamiento ─────────────────────────────────────────────
    def train(
        self,
        target_col: str,
        hidden_sizes: List[int] = None,
        epochs: int = DEFAULT_EPOCHS,
        drop_cols: Optional[list] = None,
        activation: str = "relu",
        progress_callback=None,
    ) -> dict:
        hidden_sizes = hidden_sizes or list(DEFAULT_LAYERS)
        self.target_col = target_col

        X, y = self.proc.preprocess(target_col, drop_cols=drop_cols)

        # Cachear datos procesados para feature_importance (NO re-fitear scalers)
        self._X_cached = X.copy()
        self._y_cached = y.copy()

        # ── Split temporal (NO aleatorio) ────────────────────────
        # Para series temporales: primero 80% = train, último 20% = validación
        split_idx = int(len(X) * 0.8)
        X_tr, X_val = X[:split_idx], X[split_idx:]
        y_tr, y_val = y[:split_idx], y[split_idx:]

        # ── Detección de data leakage ────────────────────────────
        self._check_leakage(X, y)

        model = NeuralNet(
            X.shape[1], hidden_sizes, output_size=1,
            task="regression", activation=activation,
        )
        self.trainer = Trainer(model)
        history = self.trainer.train(
            X_tr, y_tr, X_val, y_val,
            epochs=epochs,
            batch_size=min(DEFAULT_BATCH_SIZE, max(8, len(X_tr) // 10)),
            progress_callback=progress_callback,
        )

        # ── Métricas en escala original ──────────────────────────
        preds_s = self.trainer.predict(X_val)
        y_real = self.proc.inverse_target(y_val)
        p_real = self.proc.inverse_target(preds_s)

        self.metrics = {
            "MAE":      round(mean_absolute_error(y_real, p_real), 4),
            "RMSE":     round(math.sqrt(mean_squared_error(y_real, p_real)), 4),
            "R²":       round(r2_score(y_real, p_real), 4),
            "MAPE (%)": round(float(np.mean(np.abs((y_real - p_real) / (np.abs(y_real) + 1e-8))) * 100), 4),
            "samples":  len(X),
            "features": self.proc.feature_names,
            "epochs_run": self.trainer.stopped_epoch,
            "history":  history,
        }
        return self.metrics

    # ── Predicción sobre datos nuevos ────────────────────────────
    def predict_df(self, df: pd.DataFrame) -> np.ndarray:
        """Predice sobre un DataFrame externo (mismas columnas que el entrenamiento)."""
        if self.trainer is None:
            raise RuntimeError("Entrena el modelo primero con .train()")
        X = self.proc.scaler_X.transform(
            df[self.proc.feature_names].fillna(0).values.astype(np.float32)
        )
        preds_s = self.trainer.predict(X)
        return self.proc.inverse_target(preds_s)

    # ── Pronóstico futuro (serie temporal) ───────────────────────
    def forecast(self, periods: int = 5) -> pd.DataFrame:
        """
        Genera pronóstico recursivo para los próximos N períodos.
        Incluye intervalos de confianza calculados con bootstrap.
        """
        if self.trainer is None:
            raise RuntimeError("Entrena el modelo primero con .train()")

        X = self._X_cached
        y = self._y_cached
        target = self.target_col
        history = self.proc.df[target].dropna().values.tolist()

        # Índices de lag features y rolling features
        lag_idxs = [i for i, c in enumerate(self.proc.feature_names)
                    if c.startswith(f"{target}_lag")]
        rolling_idxs = [i for i, c in enumerate(self.proc.feature_names)
                        if "media_movil" in c or "std_movil" in c]

        # Semilla: última fila procesada
        last_row_scaled = X[-1].copy()
        lag_window_scaled = list(y[-5:])

        results = []
        prev_real = history[-1] if history else 0.0

        for step in range(1, periods + 1):
            row = last_row_scaled.copy()

            # Actualizar columnas de lag con valores predichos previos
            for k, idx in enumerate(sorted(lag_idxs)):
                lag_pos = k + 1
                val_s = lag_window_scaled[-lag_pos] if lag_pos <= len(lag_window_scaled) else 0.0
                row[idx] = float(val_s)

            # Actualizar rolling features
            if len(lag_window_scaled) >= 3:
                for idx in rolling_idxs:
                    fname = self.proc.feature_names[idx]
                    if "media_movil" in fname:
                        row[idx] = float(np.mean(lag_window_scaled[-3:]))
                    elif "std_movil" in fname:
                        row[idx] = float(np.std(lag_window_scaled[-3:]))

            X_row = row.reshape(1, -1).astype(np.float32)
            pred_s = self.trainer.predict(X_row)[0]
            pred_real = float(self.proc.inverse_target(np.array([pred_s]))[0])

            # ── Bootstrap para intervalo de confianza ────────────
            lower, upper = self._bootstrap_confidence(row, step)

            var_pct = ((pred_real - prev_real) / (abs(prev_real) + 1e-8)) * 100
            conf_label = "Alta" if step <= 2 else "Media" if step <= 4 else "Baja"

            results.append({
                "periodo":       f"T+{step}",
                "prediccion":    round(pred_real, 2),
                "intervalo_inf": round(lower, 2),
                "intervalo_sup": round(upper, 2),
                "variacion_pct": round(var_pct, 2),
                "confianza":     conf_label,
            })

            lag_window_scaled.append(pred_s)
            prev_real = pred_real

        return pd.DataFrame(results)

    def _bootstrap_confidence(self, row: np.ndarray, step: int) -> tuple:
        """Calcula intervalo de confianza con bootstrap (percentiles 5-95)."""
        predictions = []
        noise_scale = CONFIDENCE_NOISE_STD * step  # Más incertidumbre a futuro

        for _ in range(CONFIDENCE_BOOTSTRAP_N):
            noisy_row = row.copy()
            noise = np.random.normal(0, noise_scale, size=row.shape)
            noisy_row = noisy_row + noise
            noisy_row = np.clip(noisy_row, 0, 1)  # Mantener en rango [0,1]
            X_noisy = noisy_row.reshape(1, -1).astype(np.float32)
            pred_s = self.trainer.predict(X_noisy)[0]
            pred_real = float(self.proc.inverse_target(np.array([pred_s]))[0])
            predictions.append(pred_real)

        return float(np.percentile(predictions, 5)), float(np.percentile(predictions, 95))

    # ── Importancia de features (SIN re-fitear scalers) ──────────
    def feature_importance(self) -> dict:
        """
        Calcula importancia de variables por permutación.
        USA DATOS CACHEADOS — no llama preprocess() de nuevo.
        """
        if self.trainer is None:
            return {}
        if self._X_cached is None or self._y_cached is None:
            return {}

        X = self._X_cached
        y = self._y_cached
        base = math.sqrt(mean_squared_error(y, self.trainer.predict(X)))

        scores = {}
        for i, feat in enumerate(self.proc.feature_names):
            Xp = X.copy()
            np.random.shuffle(Xp[:, i])
            err = math.sqrt(mean_squared_error(y, self.trainer.predict(Xp)))
            scores[feat] = round(max(0.0, err - base), 6)
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    # ── Exportar pronóstico ──────────────────────────────────────
    def export_forecast(self, forecast_df: pd.DataFrame, path: str = "predicciones.csv"):
        """Exporta el pronóstico a un archivo CSV."""
        forecast_df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    # ── Detección de data leakage ────────────────────────────────
    def _check_leakage(self, X: np.ndarray, y: np.ndarray):
        """Alerta si algún feature tiene correlación > 0.99 con el target."""
        for i, feat in enumerate(self.proc.feature_names):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            if abs(corr) > 0.99:
                import warnings
                warnings.warn(
                    f"⚠ Posible data leakage: '{feat}' tiene correlación "
                    f"{corr:.3f} con el target. Considera eliminarlo.",
                    UserWarning,
                )

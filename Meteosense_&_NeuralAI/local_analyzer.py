"""
local_analyzer.py
=================
Motor de análisis inteligente 100% local — sin API key.

Detecta automáticamente:
  · Tendencias (creciente / decreciente / estable / cíclica)
  · Estacionalidad y patrones periódicos
  · Correlaciones entre variables
  · Anomalías y valores atípicos
  · Distribuciones (normal, sesgada, bimodal)
  · Importancia de variables

Genera explicaciones en español y planes de acción
usando reglas estadísticas, sin IA externa.
"""

import math
import hashlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from typing import Optional

from config import (
    CORRELATION_THRESHOLD, OUTLIER_IQR_MULTIPLIER,
    TREND_R2_MIN, SEASONALITY_ACF_HEIGHT, SEASONALITY_MAX_LAG_RATIO,
)


# ─────────────────────────────────────────────────────────────────
# Utilidades estadísticas
# ─────────────────────────────────────────────────────────────────

def _trend_slope(values: np.ndarray) -> float:
    """Pendiente normalizada de la regresión lineal."""
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    mean = np.mean(values)
    # Protección contra división por cero o media muy pequeña
    if abs(mean) < 1e-8:
        return 0.0
    return slope / abs(mean) * 100  # % de cambio por período

def _r_squared(values: np.ndarray) -> float:
    x = np.arange(len(values))
    slope, intercept = np.polyfit(x, values, 1)
    fitted = slope * x + intercept
    ss_res = np.sum((values - fitted) ** 2)
    ss_tot = np.sum((values - np.mean(values)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0

def _cv(values: np.ndarray) -> float:
    """Coeficiente de variación (volatilidad)."""
    m = np.mean(values)
    return (np.std(values) / abs(m) * 100) if abs(m) > 1e-8 else 0

def _skewness_label(sk: float) -> str:
    if abs(sk) < 0.5:   return "simétrica"
    if sk > 1.5:        return "muy sesgada a la derecha"
    if sk > 0.5:        return "sesgada a la derecha"
    if sk < -1.5:       return "muy sesgada a la izquierda"
    return "sesgada a la izquierda"

def _trend_label(slope_pct: float, r2: float) -> str:
    if r2 < TREND_R2_MIN:
        return "sin tendencia clara"
    if abs(slope_pct) < 0.5:
        return "estable"
    if slope_pct > 5:   return "crecimiento fuerte"
    if slope_pct > 1:   return "crecimiento moderado"
    if slope_pct > 0:   return "leve crecimiento"
    if slope_pct < -5:  return "caída fuerte"
    if slope_pct < -1:  return "caída moderada"
    return "leve caída"

def _detect_seasonality(values: np.ndarray) -> Optional[int]:
    """Detecta si hay un patrón periódico y retorna su período."""
    if len(values) < 12:
        return None
    max_lag = min(len(values) // 2, max(52, int(len(values) * SEASONALITY_MAX_LAG_RATIO)))
    acf = []
    for lag in range(1, max_lag):
        try:
            r = np.corrcoef(values[:-lag], values[lag:])[0, 1]
            acf.append(r if not np.isnan(r) else 0)
        except Exception:
            acf.append(0)
    if not acf:
        return None
    peaks, _ = find_peaks(acf, height=SEASONALITY_ACF_HEIGHT, distance=3)
    if len(peaks):
        return int(peaks[0]) + 1
    return None

def _outlier_mask(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    m = OUTLIER_IQR_MULTIPLIER
    return (s < q1 - m * iqr) | (s > q3 + m * iqr)

def _fmt(v: float, currency: str = "") -> str:
    if abs(v) >= 1_000_000:
        return f"{currency}{v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{currency}{v/1_000:.1f}K"
    return f"{currency}{v:.2f}"

def _is_normal(values: np.ndarray) -> tuple:
    """Test de normalidad Shapiro-Wilk. Retorna (es_normal, p_value)."""
    if len(values) < 8 or len(values) > 5000:
        return None, None
    try:
        stat, p = stats.shapiro(values[:5000])
        return p > 0.05, round(p, 4)
    except Exception:
        return None, None


# ─────────────────────────────────────────────────────────────────
# Analizador principal
# ─────────────────────────────────────────────────────────────────

class LocalAnalyzer:
    """
    Analiza un DataFrame y genera explicaciones y recomendaciones
    en español sin ninguna API externa.
    Incluye caché para evitar recálculos innecesarios.
    """

    def __init__(self, df: pd.DataFrame):
        self.df  = df
        self.num = df.select_dtypes(include=np.number)
        self.cat = df.select_dtypes(exclude=np.number)
        self._cache: dict = {}
        self._cache_hash: str = ""

    def _df_hash(self) -> str:
        """Hash rápido del DataFrame para caché."""
        try:
            return hashlib.md5(
                pd.util.hash_pandas_object(self.df).values.tobytes()
            ).hexdigest()
        except Exception:
            return ""

    def _get_cached(self, key: str):
        """Retorna resultado cacheado si el DataFrame no ha cambiado."""
        current_hash = self._df_hash()
        if current_hash == self._cache_hash and key in self._cache:
            return self._cache[key]
        if current_hash != self._cache_hash:
            self._cache.clear()
            self._cache_hash = current_hash
        return None

    def _set_cached(self, key: str, value):
        self._cache[key] = value
        return value

    # ── Análisis completo del dataset ─────────────────────────────
    def analyze_dataset(self) -> dict:
        cached = self._get_cached("dataset_analysis")
        if cached:
            return cached

        resumen       = self._resumen()
        columnas      = self._columnas_detalle()
        correlaciones = self._correlaciones()
        anomalias     = self._anomalias()
        tendencias    = self._tendencias()

        # Pasar resultados ya calculados al texto (evitar recálculo)
        texto    = self._texto_dataset(resumen, columnas, correlaciones, anomalias)
        acciones = self._acciones_dataset(resumen, columnas, correlaciones, anomalias)

        result = {
            "resumen":       resumen,
            "columnas":      columnas,
            "correlaciones": correlaciones,
            "anomalias":     anomalias,
            "tendencias":    tendencias,
            "texto":         texto,
            "acciones":      acciones,
        }
        return self._set_cached("dataset_analysis", result)

    # ── Análisis de predicciones ──────────────────────────────────
    def analyze_predictions(
        self,
        target_col: str,
        metrics: dict,
        forecast_df: pd.DataFrame,
        importance: dict,
    ) -> dict:
        result = {}
        result["calidad_modelo"]  = self._calidad_modelo(metrics)
        result["interpretacion"]  = self._interpretar_forecast(target_col, forecast_df)
        result["variables_clave"] = self._variables_clave(importance)
        result["texto"]           = self._texto_predicciones(target_col, metrics, forecast_df, importance)
        result["plan_accion"]     = self._plan_accion(target_col, metrics, forecast_df, importance)
        result["riesgos"]         = self._riesgos(metrics, forecast_df)
        return result

    # ─────────────────────────────────────────────────────────────
    # Secciones internas — Dataset
    # ─────────────────────────────────────────────────────────────

    def _resumen(self) -> dict:
        return {
            "filas": len(self.df),
            "columnas": len(self.df.columns),
            "numericas": len(self.num.columns),
            "categoricas": len(self.cat.columns),
            "nulos_total": int(self.df.isnull().sum().sum()),
            "duplicados": int(self.df.duplicated().sum()),
        }

    def _columnas_detalle(self) -> list:
        detalles = []
        for col in self.num.columns:
            s = self.num[col].dropna()
            if len(s) < 2:
                continue
            vals = s.values
            slope = _trend_slope(vals)
            r2    = _r_squared(vals)
            sk    = float(stats.skew(vals))
            kurt  = float(stats.kurtosis(vals))
            season = _detect_seasonality(vals)
            normal, p_normal = _is_normal(vals)
            detalles.append({
                "columna":        col,
                "media":          round(float(s.mean()), 4),
                "mediana":        round(float(s.median()), 4),
                "std":            round(float(s.std()), 4),
                "min":            round(float(s.min()), 4),
                "max":            round(float(s.max()), 4),
                "cv_pct":         round(_cv(vals), 2),
                "tendencia":      _trend_label(slope, r2),
                "slope_pct":      round(slope, 4),
                "r2_tendencia":   round(r2, 4),
                "asimetria":      _skewness_label(sk),
                "skewness":       round(sk, 4),
                "kurtosis":       round(kurt, 4),
                "estacionalidad": season,
                "outliers":       int(_outlier_mask(s).sum()),
                "es_normal":      normal,
                "p_normalidad":   p_normal,
            })
        return detalles

    def _correlaciones(self) -> list:
        if len(self.num.columns) < 2:
            return []
        corr = self.num.corr()
        pares = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iloc[i, j]
                if not math.isnan(r) and abs(r) > CORRELATION_THRESHOLD:
                    pares.append({
                        "col_a": cols[i],
                        "col_b": cols[j],
                        "r":     round(r, 4),
                        "tipo":  "positiva" if r > 0 else "negativa",
                        "fuerza": "muy fuerte" if abs(r) > 0.85
                                  else "fuerte" if abs(r) > 0.65
                                  else "moderada",
                    })
        return sorted(pares, key=lambda x: abs(x["r"]), reverse=True)

    def _anomalias(self) -> list:
        resultado = []
        for col in self.num.columns:
            s = self.df[col].dropna()
            mask = _outlier_mask(s)
            n = mask.sum()
            if n > 0:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                m = OUTLIER_IQR_MULTIPLIER
                resultado.append({
                    "columna":  col,
                    "cantidad": int(n),
                    "pct":      round(n / len(s) * 100, 2),
                    "rango_ok": f"{q1 - m*iqr:.2f} – {q3 + m*iqr:.2f}",
                    "valores_atipicos": s[mask].head(5).round(2).tolist(),
                })
        return resultado

    def _tendencias(self) -> list:
        result = []
        for col in self.num.columns:
            s = self.df[col].dropna()
            if len(s) < 4:
                continue
            v = s.values
            slope  = _trend_slope(v)
            r2     = _r_squared(v)
            season = _detect_seasonality(v)
            crecimiento = (v[-1] - v[0]) / (abs(v[0]) + 1e-8) * 100
            result.append({
                "columna":              col,
                "tendencia":            _trend_label(slope, r2),
                "crecimiento_total_pct": round(crecimiento, 2),
                "volatilidad":   "alta" if _cv(v) > 30 else "media" if _cv(v) > 10 else "baja",
                "estacionalidad": f"cada {season} períodos" if season else "no detectada",
            })
        return result

    # ─────────────────────────────────────────────────────────────
    # Generación de texto — Dataset (recibe datos precalculados)
    # ─────────────────────────────────────────────────────────────

    def _texto_dataset(self, r=None, det=None, cor=None, ano=None) -> str:
        if r is None:
            r = self._resumen()
        if det is None:
            det = self._columnas_detalle()
        if cor is None:
            cor = self._correlaciones()
        if ano is None:
            ano = self._anomalias()

        lineas = []
        lineas.append(
            f"El dataset contiene {r['filas']:,} registros con {r['columnas']} variables "
            f"({r['numericas']} numéricas y {r['categoricas']} categóricas)."
        )

        if r["nulos_total"] > 0:
            pct = r["nulos_total"] / (r["filas"] * r["columnas"]) * 100
            nivel = "alto" if pct > 10 else "moderado" if pct > 3 else "bajo"
            lineas.append(
                f"Se detectaron {r['nulos_total']:,} valores nulos "
                f"({pct:.1f}% del total) — nivel {nivel}."
            )
        else:
            lineas.append("El dataset no tiene valores nulos — datos completos.")

        # Tendencias de columnas numéricas
        for d in det:
            cv = d["cv_pct"]
            vol = "muy volátil" if cv > 50 else "volátil" if cv > 25 else "estable"
            texto_col = (
                f"La variable '{d['columna']}' tiene tendencia {d['tendencia']} "
                f"(R²={d['r2_tendencia']:.2f}) y distribución {d['asimetria']}. "
                f"Su comportamiento es {vol} (CV={cv:.1f}%)."
            )
            # FIX: comparar con None real, no con string "None"
            if d["estacionalidad"] is not None:
                texto_col += f" Se detecta estacionalidad cada {d['estacionalidad']} períodos."
            if d.get("es_normal") is not None:
                dist_tipo = "normal (Shapiro p={:.3f})".format(d["p_normalidad"]) if d["es_normal"] \
                           else "no normal (Shapiro p={:.3f})".format(d["p_normalidad"])
                texto_col += f" Distribución {dist_tipo}."
            lineas.append(texto_col)

        # Correlaciones relevantes
        if cor:
            for c in cor[:3]:
                direccion = "cuando una sube, la otra también sube" if c["tipo"] == "positiva" \
                            else "cuando una sube, la otra baja"
                lineas.append(
                    f"Correlación {c['fuerza']} {c['tipo']} entre "
                    f"'{c['col_a']}' y '{c['col_b']}' (r={c['r']:.2f}): {direccion}."
                )

        # Anomalías
        if ano:
            cols_con_outliers = [f"'{a['columna']}' ({a['cantidad']} valores, {a['pct']}%)"
                                 for a in ano[:3]]
            lineas.append(
                "Se detectaron valores atípicos en: " + ", ".join(cols_con_outliers) + "."
            )

        return " ".join(lineas)

    def _acciones_dataset(self, r=None, det=None, cor=None, ano=None) -> list:
        if r is None:
            r = self._resumen()
        if det is None:
            det = self._columnas_detalle()
        if cor is None:
            cor = self._correlaciones()
        if ano is None:
            ano = self._anomalias()

        acciones = []

        if r["nulos_total"] > 0:
            acciones.append({
                "prioridad": "Alta",
                "accion": "Tratar valores nulos",
                "detalle": f"Hay {r['nulos_total']} nulos. Decide si imputar con media/mediana o eliminar filas.",
            })

        if r["duplicados"] > 0:
            acciones.append({
                "prioridad": "Alta",
                "accion": "Eliminar duplicados",
                "detalle": f"Se encontraron {r['duplicados']} filas duplicadas que pueden distorsionar el modelo.",
            })

        for a in ano:
            if a["pct"] > 5:
                acciones.append({
                    "prioridad": "Media",
                    "accion": f"Revisar outliers en '{a['columna']}'",
                    "detalle": f"{a['pct']:.1f}% de registros fuera del rango normal. "
                               f"Verifica si son errores de captura o eventos reales.",
                })

        for d in det:
            if d["cv_pct"] > 60:
                acciones.append({
                    "prioridad": "Media",
                    "accion": f"Normalizar '{d['columna']}'",
                    "detalle": f"Alta volatilidad (CV={d['cv_pct']:.1f}%). "
                               "Considera transformación logarítmica para estabilizar.",
                })

        fuertes = [c for c in cor if abs(c["r"]) > 0.85]
        if fuertes:
            par = fuertes[0]
            acciones.append({
                "prioridad": "Media",
                "accion": "Revisar multicolinealidad",
                "detalle": f"'{par['col_a']}' y '{par['col_b']}' tienen correlación {par['r']:.2f}. "
                           "Considera eliminar una para evitar redundancia en el modelo.",
            })

        for d in det:
            if "caída" in d["tendencia"] and d["r2_tendencia"] > 0.3:
                acciones.append({
                    "prioridad": "Alta",
                    "accion": f"Atención: '{d['columna']}' en declive",
                    "detalle": f"Tendencia de {d['tendencia']} con R²={d['r2_tendencia']:.2f}. "
                               "Investiga causas y toma acciones correctivas.",
                })

        if not acciones:
            acciones.append({
                "prioridad": "Baja",
                "accion": "Dataset en buenas condiciones",
                "detalle": "No se detectaron problemas críticos. Puedes proceder al entrenamiento.",
            })

        return acciones

    # ─────────────────────────────────────────────────────────────
    # Secciones internas — Predicciones
    # ─────────────────────────────────────────────────────────────

    def _calidad_modelo(self, metrics: dict) -> dict:
        r2 = metrics.get("R²", 0)
        mape = metrics.get("MAPE (%)", 100)

        if r2 >= 0.90:   calidad = "Excelente"
        elif r2 >= 0.75: calidad = "Buena"
        elif r2 >= 0.55: calidad = "Aceptable"
        else:            calidad = "Baja — usar con precaución"

        if mape <= 5:    precision = "muy alta (error < 5%)"
        elif mape <= 10: precision = "alta (error < 10%)"
        elif mape <= 20: precision = "moderada (error < 20%)"
        else:            precision = f"baja (error promedio {mape:.1f}%)"

        return {"nivel": calidad, "precision": precision, "r2": r2, "mape": mape}

    def _interpretar_forecast(self, target_col: str, fc: pd.DataFrame) -> dict:
        preds = fc["prediccion"].values
        vars_ = fc["variacion_pct"].values

        tendencia_fc = "creciente"  if np.mean(vars_) > 1   else \
                       "decreciente" if np.mean(vars_) < -1  else "estable"
        mejor_idx  = int(np.argmax(preds))
        peor_idx   = int(np.argmin(preds))
        crecimiento_total = float((preds[-1] - preds[0]) / (abs(preds[0]) + 1e-8) * 100)

        return {
            "tendencia":          tendencia_fc,
            "mejor_periodo":      fc.iloc[mejor_idx]["periodo"],
            "mejor_valor":        round(float(preds[mejor_idx]), 2),
            "peor_periodo":       fc.iloc[peor_idx]["periodo"],
            "crecimiento_total":  round(crecimiento_total, 2),
            "variacion_promedio": round(float(np.mean(vars_)), 2),
            "total_acumulado":    round(float(preds.sum()), 2),
        }

    def _variables_clave(self, importance: dict) -> list:
        if not importance:
            return []
        total = sum(importance.values()) + 1e-9
        return [
            {
                "variable": k,
                "score": round(v, 6),
                "pct_influencia": round(v / total * 100, 1),
            }
            for k, v in list(importance.items())[:6]
        ]

    # ─────────────────────────────────────────────────────────────
    # Generación de texto — Predicciones
    # ─────────────────────────────────────────────────────────────

    def _texto_predicciones(
        self, target_col: str, metrics: dict, fc: pd.DataFrame, importance: dict
    ) -> str:
        cal  = self._calidad_modelo(metrics)
        interp = self._interpretar_forecast(target_col, fc)
        vk   = self._variables_clave(importance)

        lineas = []

        lineas.append(
            f"El modelo predictivo para '{target_col}' tiene calidad {cal['nivel']} "
            f"(R²={cal['r2']:.3f}), con precisión {cal['precision']}."
        )

        emojis = {"creciente": "📈", "decreciente": "📉", "estable": "➡️"}
        e = emojis.get(interp["tendencia"], "")
        lineas.append(
            f"{e} El pronóstico muestra una tendencia {interp['tendencia']} "
            f"con una variación promedio de {interp['variacion_promedio']:+.1f}% por período. "
            f"El valor acumulado proyectado es {_fmt(interp['total_acumulado'])}."
        )

        lineas.append(
            f"El período más alto es {interp['mejor_periodo']} "
            f"con {_fmt(interp['mejor_valor'])}, "
            f"y el crecimiento total proyectado es {interp['crecimiento_total']:+.1f}%."
        )

        if vk:
            top3 = vk[:3]
            vars_str = ", ".join(
                f"'{v['variable']}' ({v['pct_influencia']:.0f}%)" for v in top3
            )
            lineas.append(
                f"Las variables que más influyen en la predicción son: {vars_str}."
            )

        if metrics.get("epochs_run"):
            lineas.append(
                f"El modelo se entrenó en {metrics['epochs_run']} épocas "
                f"con {metrics.get('samples', '?')} muestras."
            )

        return " ".join(lineas)

    def _plan_accion(
        self, target_col: str, metrics: dict, fc: pd.DataFrame, importance: dict
    ) -> list:
        acciones = []
        cal    = self._calidad_modelo(metrics)
        interp = self._interpretar_forecast(target_col, fc)
        vk     = self._variables_clave(importance)

        # 1. Calidad del modelo (términos no técnicos)
        if cal["r2"] < 0.6:
            acciones.append({
                "orden": 1,
                "prioridad": "Media",
                "accion": "Pronóstico con alta incertidumbre",
                "detalle": "Los datos actuales son muy variables. Considera estas predicciones como una referencia general y mantén monitoreo constante."
            })
        else:
            acciones.append({
                "orden": 1,
                "prioridad": "Baja",
                "accion": "Pronóstico confiable",
                "detalle": "El sistema ha encontrado patrones estables. Las predicciones tienen buena fiabilidad para entender cómo se comportará el entorno."
            })

        # 2. Recomendaciones en tiempo real basadas en la variable y tendencia
        var_lower = target_col.lower()
        tendencia = interp["tendencia"]
        promedio_fc = fc["prediccion"].mean() if not fc.empty else 0

        if "temp" in var_lower:
            if promedio_fc > 30 or tendencia == "creciente":
                acciones.append({
                    "orden": 2,
                    "prioridad": "Alta",
                    "accion": "Precaución por altas temperaturas",
                    "detalle": f"Se prevé una tendencia al alza o calor intenso (~{promedio_fc:.1f}°). Recomendable mantenerse hidratado, proteger equipos sensibles al calor y evitar exposición prolongada al sol."
                })
            elif promedio_fc < 12 or tendencia == "decreciente":
                acciones.append({
                    "orden": 2,
                    "prioridad": "Media",
                    "accion": "Descenso de temperatura detectado",
                    "detalle": f"El pronóstico indica un enfriamiento del ambiente (~{promedio_fc:.1f}°). Considera resguardar áreas vulnerables al frío."
                })
            else:
                acciones.append({
                    "orden": 2,
                    "prioridad": "Baja",
                    "accion": "Condiciones térmicas estables",
                    "detalle": "Las temperaturas proyectadas se encuentran dentro de un rango confortable y normal para la estación."
                })
        elif "hum" in var_lower:
            if promedio_fc > 75 or tendencia == "creciente":
                acciones.append({
                    "orden": 2,
                    "prioridad": "Media",
                    "accion": "Alta humedad en el ambiente",
                    "detalle": "El exceso de humedad puede provocar sensación bochornosa y aumentar el riesgo de precipitaciones. Verifica la ventilación."
                })
            elif promedio_fc < 30:
                acciones.append({
                    "orden": 2,
                    "prioridad": "Media",
                    "accion": "Ambiente inusualmente seco",
                    "detalle": "Baja humedad detectada. Aumenta el riesgo de resequedad. Se sugiere uso de humectadores en áreas cerradas."
                })
        elif "uv" in var_lower or "luz" in var_lower:
            if promedio_fc > 7 or tendencia == "creciente":
                acciones.append({
                    "orden": 2,
                    "prioridad": "Alta",
                    "accion": "Alerta por radiación UV",
                    "detalle": "Se esperan picos altos de luz solar/UV. Indispensable usar bloqueador solar y limitar el trabajo pesado al aire libre al mediodía."
                })
        elif "pm" in var_lower or "aire" in var_lower:
            if promedio_fc > 35 or tendencia == "creciente":
                acciones.append({
                    "orden": 2,
                    "prioridad": "Urgente",
                    "accion": "Deterioro en calidad del aire",
                    "detalle": "Las partículas suspendidas están subiendo. Limita las actividades físicas al aire libre y usa mascarilla si tienes sensibilidad respiratoria."
                })
        else:
            if tendencia == "creciente":
                acciones.append({
                    "orden": 2,
                    "prioridad": "Media",
                    "accion": f"Aumento progresivo en {target_col}",
                    "detalle": "La variable analizada está mostrando un patrón de incremento sostenido para las próximas horas."
                })
            elif tendencia == "decreciente":
                acciones.append({
                    "orden": 2,
                    "prioridad": "Media",
                    "accion": f"Disminución progresiva en {target_col}",
                    "detalle": "Se proyecta una caída constante de este indicador. Vigila si esto representa algún impacto para tu entorno."
                })
            else:
                acciones.append({
                    "orden": 2,
                    "prioridad": "Baja",
                    "accion": f"Estabilidad en {target_col}",
                    "detalle": "No se detectan sobresaltos importantes. El entorno físico se mantendrá equilibrado."
                })

        # 3. Importancia de Variables de manera amigable
        if vk:
            top = vk[0]
            if top['variable'] != target_col:
                acciones.append({
                    "orden": 3,
                    "prioridad": "Baja",
                    "accion": "Dato curioso detectado por la IA",
                    "detalle": f"El comportamiento actual de '{target_col}' está siendo fuertemente influenciado por los cambios en '{top['variable']}'. Si esta variable cambia, impactará en el pronóstico."
                })

        # 4. Confianza a largo plazo
        conf_bajas = fc[fc["confianza"] == "Baja"]
        if len(conf_bajas):
            acciones.append({
                "orden": 4,
                "prioridad": "Media",
                "accion": "Pronóstico a largo plazo",
                "detalle": f"Las predicciones más lejanas ({len(conf_bajas)} períodos) son menos seguras. Te sugerimos darle más peso a los datos iniciales de la gráfica."
            })

        return sorted(acciones, key=lambda x: x["orden"])

    def _riesgos(self, metrics: dict, fc: pd.DataFrame) -> list:
        riesgos = []
        vars_ = fc["variacion_pct"].values

        if metrics.get("MAPE (%)", 0) > 15:
            riesgos.append({
                "nivel": "Alto",
                "riesgo": "Imprecisión del modelo",
                "descripcion": f"Error promedio del {metrics['MAPE (%)']:.1f}%. "
                               "Las predicciones pueden diferir significativamente de la realidad.",
            })

        if np.std(vars_) > 10:
            riesgos.append({
                "nivel": "Medio",
                "riesgo": "Alta variabilidad en el pronóstico",
                "descripcion": "Las predicciones tienen oscilaciones grandes entre períodos. "
                               "Monitorea semanalmente y ajusta los planes de corto plazo.",
            })

        if len(fc) > 3 and fc.iloc[-1]["confianza"] == "Baja":
            riesgos.append({
                "nivel": "Medio",
                "riesgo": "Incertidumbre a largo plazo",
                "descripcion": "Los últimos períodos del pronóstico tienen confianza baja. "
                               "Factores externos no capturados en los datos pueden alterar el resultado.",
            })

        if not riesgos:
            riesgos.append({
                "nivel": "Bajo",
                "riesgo": "Sin riesgos críticos identificados",
                "descripcion": "El modelo opera dentro de parámetros aceptables.",
            })

        return riesgos

    # ─────────────────────────────────────────────────────────────
    # Datos para gráficas
    # ─────────────────────────────────────────────────────────────

    def chart_data(self) -> dict:
        """Retorna todos los datos necesarios para las gráficas del reporte."""
        corr_data = {}
        if len(self.num.columns) >= 2:
            cm = self.num.corr().round(3)
            corr_data = {
                "cols": list(cm.columns),
                "matrix": cm.values.tolist(),
            }

        distributions = {}
        for col in self.num.columns:
            s = self.num[col].dropna()
            hist, edges = np.histogram(s, bins=20)
            distributions[col] = {
                "counts": hist.tolist(),
                "edges":  [round(e, 2) for e in edges.tolist()],
            }

        # Datos de TODAS las columnas numéricas para serie histórica interactiva
        all_series = {}
        for col in self.num.columns:
            vals = self.num[col].dropna().tolist()
            all_series[col] = vals

        return {
            "correlations": corr_data,
            "distributions": distributions,
            "all_series": all_series,
        }

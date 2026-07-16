"""
config.py — Neural AI v3.1 (Meteo)
==================================
Configuración del sistema meteorológico y alertas.
"""

VERSION = "3.1.0"

# =====================================================================
# CONFIGURACIÓN DE RED NEURONAL (PYTORCH)
# =====================================================================
DEFAULT_LR = 0.001
DEFAULT_DROPOUT = 0.2
EARLY_STOPPING_PATIENCE = 20
DEFAULT_ACTIVATION = "relu"

DEFAULT_LAYERS = [64, 32]
DEFAULT_EPOCHS = 80
DEFAULT_BATCH_SIZE = 32

# =====================================================================
# CONFIGURACIÓN DEL ANALIZADOR ESTADÍSTICO
# =====================================================================
CORRELATION_THRESHOLD = 0.5
OUTLIER_IQR_MULTIPLIER = 1.5
TREND_R2_MIN = 0.2
SEASONALITY_ACF_HEIGHT = 0.2
SEASONALITY_MAX_LAG_RATIO = 0.3

MIN_ROWS_FOR_TRAINING = 20
CONFIDENCE_BOOTSTRAP_N = 50
CONFIDENCE_NOISE_STD = 0.02

# =====================================================================
# CONFIGURACIÓN METEOROLÓGICA (THINGSPEAK)
# =====================================================================
# Canales ThingSpeak a monitorear
EXAMPLE_CHANNELS = {
    1: {"id": 3269359, "nombre": "Proyecto UNAD (Principal)", "ubicacion": "IoT"},
    2: {"id": 12397,   "nombre": "Estación Meteorológica USA", "ubicacion": "USA"},
    3: {"id": 9,       "nombre": "Sensor IoT Público", "ubicacion": "Global"},
    4: {"id": 2610851, "nombre": "Aquarium Water and Temp", "ubicacion": "Acuario"},
    5: {"id": 1785844, "nombre": "Wind Power Smart Monitor", "ubicacion": "Generador"},
    6: {"id": 2862014, "nombre": "Power Grid Smart Monitor", "ubicacion": "Red Eléctrica"}
}

# Umbrales para alertas automáticas
ALERT_THRESHOLDS = {
    "temperatura": {"max": 35.0, "min": -5.0, "label": "°C"},
    "humedad":     {"max": 85.0, "min": 15.0, "label": "%"},
    "pm25":        {"max": 50.0, "min": 0.0,  "label": "µg/m³"},
    "uv":          {"max": 8.0,  "min": 0.0,  "label": "Índice UV"}
}

# Constantes de ThingSpeak
THINGSPEAK_BASE_URL = "https://api.thingspeak.com"

# Auto-descubrimiento de campos
FIELD_KEYWORDS = {
    "temperatura": ["temp", "temperatura", "celsius", "farenheit"],
    "humedad": ["hum", "humidity", "humedad"],
    "uv": ["uv", "ultra", "luz"],
    "pm25": ["pm", "pm25", "pm2.5", "polvo", "aire"],
    "lluvia": ["rain", "pluvio", "lluvia"],
    "presion": ["press", "presion", "hpa"],
}

# Rangos físicos válidos
VALID_RANGES = {
    "temperatura": (-20, 60),
    "humedad": (0, 100),
    "uv": (0, 15),
    "pm25": (0, 1000),
    "lluvia": (0, 500)
}

"""
app.py
======
Servidor Backend local con Flask para Neural AI v3.
Provee la API y sirve el Dashboard Web interactivo.
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np

# Componentes de Neural AI
from station_manager import StationManager
from processor import DataProcessor
from predictor import Predictor
from local_analyzer import LocalAnalyzer

app = Flask(__name__)

# Instancia global del gestor de estaciones
sm = StationManager()

# Variables para mantener estado de la IA
current_predictor = None
current_processor = None
current_analyzer = None
current_df = None

@app.route("/")
def index():
    """Renderiza el Dashboard Web."""
    return render_template("index.html")

@app.route("/api/stations", methods=["GET"])
def get_stations():
    """Retorna la lista de todas las estaciones."""
    return jsonify(sm.station_summary())

@app.route("/api/station/<int:sid>", methods=["GET"])
def get_station_data(sid):
    """Carga y retorna los datos recientes de una estación."""
    global current_df, current_processor, current_analyzer
    
    # Cargar 8000 registros
    df = sm.load_station(sid, results=8000)
    if df is None:
        return jsonify({"error": f"No se pudo conectar a la estación {sid}"}), 500
        
    current_df = df
    current_processor = DataProcessor()
    current_processor.set_data(df)
    current_analyzer = LocalAnalyzer(df)
    
    # Extraer últimos datos
    last_val = sm.stations[sid].last_value()
    
    # Formatear datos históricos para la gráfica (últimos 50 puntos para no saturar)
    # Seleccionamos un par de campos numéricos
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    history = {}
    
    # Reducimos puntos para que la gráfica en frontend sea rápida (ej: 100 puntos)
    n = len(df)
    step = max(1, n // 100)
    df_chart = df.iloc[::step].copy()
    
    if 'fecha' in df_chart.columns:
        history['labels'] = df_chart['fecha'].dt.strftime('%H:%M').tolist()
    else:
        history['labels'] = [str(i) for i in range(len(df_chart))]
        
    for col in num_cols:
        history[col] = df_chart[col].fillna(0).tolist()
        
    # Obtener alertas solo de esta estación
    todas_alertas = sm.alerts()
    alertas = [a for a in todas_alertas if a["estacion"] == sid]
    
    return jsonify({
        "info": sm.station_summary()[sid-1] if sid <= len(sm.stations) else {},
        "last_value": last_val,
        "history": history,
        "columns": num_cols,
        "alerts": alertas,
        "registros": len(df)
    })

@app.route("/api/train", methods=["POST"])
def train_and_predict():
    """Entrena la red neuronal y genera pronóstico."""
    global current_predictor, current_df, current_processor, current_analyzer
    
    if current_df is None:
        return jsonify({"error": "No hay datos cargados. Selecciona una estación primero."}), 400
        
    data = request.json
    target_col = data.get("target_col")
    epochs = int(data.get("epochs", 80))
    periods = int(data.get("periods", 5))
    
    if target_col not in current_df.columns:
        return jsonify({"error": f"Columna {target_col} no encontrada."}), 400
        
    try:
        current_predictor = Predictor(current_processor)
        metrics = current_predictor.train(target_col, epochs=epochs)
        forecast = current_predictor.forecast(periods)
        importance = current_predictor.feature_importance()
        
        # Interpretación inteligente
        pred_analysis = current_analyzer.analyze_predictions(
            target_col=target_col,
            metrics=metrics,
            forecast_df=forecast,
            importance=importance
        )
        
        return jsonify({
            "metrics": {k: v for k, v in metrics.items() if k != "history"},
            "forecast": forecast.to_dict(orient="records"),
            "importance": importance,
            "analysis": pred_analysis,
            "loss_history": metrics.get("history", {})
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

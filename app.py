import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_PATH = "best_superkart_rf_model.pkl"
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "online", "message": "SuperKart Sales Prediction API is running!"}), 200

@app.route("/v1/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df_input = pd.DataFrame([data])
        prediction = model.predict(df_input)[0]
        return jsonify({"status": "success", "predicted_sales": float(round(prediction, 2))}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/v1/predictbatch", methods=["POST"])
def predict_batch():
    try:
        data = request.get_json()
        df_input = pd.DataFrame(data)
        predictions = model.predict(df_input)
        return jsonify({"status": "success", "batch_predictions": [float(round(p, 2)) for p in predictions]}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

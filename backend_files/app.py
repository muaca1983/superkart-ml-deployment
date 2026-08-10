
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Caminho certo porque o app.py está dentro de backend_files
model = joblib.load(os.path.join('..', 'superkart_xgb_model.pkl'))
model_columns = joblib.load(os.path.join('..', 'model_columns.pkl'))

@app.route('/')
def home():
    return jsonify({'message': 'SuperKart API rodando! Use /v1/predict'})

@app.route('/v1/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        df = df.reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(df)[0]
        return jsonify({'Predicted Sales': round(float(prediction), 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)

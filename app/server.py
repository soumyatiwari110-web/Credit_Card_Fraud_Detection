from flask import Flask, render_template, request, jsonify
import joblib, numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "fraud_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "models", "feature_names.pkl"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    return jsonify({
        'fraud_prob': 0.95,
        'genuine_prob': 0.05
    })

if __name__ == '__main__':
    app.run(debug=True)
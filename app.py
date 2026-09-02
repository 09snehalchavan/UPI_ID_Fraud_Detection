from flask import Flask, request, render_template
import joblib
import numpy as np
import os
import logging

# Enable basic logging
logging.basicConfig(level=logging.INFO)

# Create Flask app
app = Flask(__name__)

# Model path (alongside your app file or adjust as needed)
MODEL_PATH = os.path.join('model', 'trained_model.pkl')

# Ensure model exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Home route
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    required_fields = [
        'Transaction_Amount',
        'Time_of_Transaction',
        'Previous_Fraudulent_Transactions',
        'Account_Age',
        'Number_of_Transactions_Last_24H'
    ]
    try:
        # Gather and validate inputs
        features = []
        for field in required_fields:
            value = request.form.get(field, '').strip()
            if value == '':
                raise ValueError(f"Missing value for '{field}'")
            try:
                num = float(value)
            except ValueError:
                raise ValueError(f"Invalid number for '{field}': '{value}'")
            features.append(num)

        logging.info(f"Received features: {features}")

        # Make prediction
        pred = model.predict([features])[0]
        result = 'Yes (Fraudulent)' if pred == 1 else 'No (Legitimate)'
        logging.info(f"Prediction result: {result}")

        return render_template('index.html', prediction_text=f'Fraud Detected: {result}')

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return render_template('index.html', prediction_text=f'Error: {e}')

# Entrypoint
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)

import os
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib


app = Flask(__name__)


model = tf.keras.models.load_model('model/my_ml_model.h5')

scaler = joblib.load('model/scaler.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json['data']
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    prediction_binary = np.where(prediction > 0.5, 1, 0)
    result = {
        'prediction': int(prediction_binary[0][0]),
        'label': 'autistic' if prediction_binary == 1 else 'non-autistic'
    }
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=port, debug=False)
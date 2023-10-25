import joblib
import numpy as np
from flask import Flask, jsonify, request
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
import os

# Cargar el modelo previamente entrenado (por motivos de eficiencia es mejor no entrenarlo con cada solicitud)
kmeans = joblib.load("model.pkl")
scaler = joblib.load('scaler.pkl')

@app.route('/predict_cluster', methods=['POST'])
def predict_cluster():
    data = request.json
    values = np.array(list(data.values())).reshape(1, -1)
    scaled_values = scaler.transform(values)
    predicted_cluster = kmeans.predict(scaled_values)
    return jsonify({'cluster': int(predicted_cluster[0])})

if __name__ == '__main__':
    app.run(debug=True)

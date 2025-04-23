import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import time
from data import preprocessing

# Cargar modelo y vocabulario
with open(os.path.join(os.path.dirname(__file__), '..', 'model', 'model.pkl'), 'rb') as f:
    classifier, vocab = pickle.load(f)

# Crear instancia Flask y configurar rutas personalizadas
app = Flask(
    __name__,
    template_folder='../web/templates',
    static_folder='../web/static'
)

# Página principal
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para el JS (porque está fuera del static_folder)
@app.route('/script.js')
def serve_script():
    return send_from_directory('../web', 'script.js')

# API para predecir sentimiento
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    data = request.get_json()

    if 'text' not in data:
        return jsonify({'error': 'El campo "text" es requerido'}), 400

    raw_text = data['text']
    clean_text = preprocessing.clean_text(raw_text)
    bow_vector = preprocessing.text_to_bow(clean_text, vocab)
    probs = classifier.predict_proba(bow_vector)
    prediction = max(probs, key=probs.get)
    elapsed_time = round((time.time() - start_time) * 1000, 2)

    return jsonify({
        'sentiment': prediction,
        'inference_time_ms': elapsed_time,
        'probabilities': probs
    })


if __name__ == '__main__':
    app.run(debug=True)

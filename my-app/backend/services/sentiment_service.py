# services/sentiment_service.py

import joblib
import os
from transformers import pipeline

# Cargar modelo y vectorizador (una sola vez)
model_path = os.path.join("models", "model_countvectorizer_nb.pkl")
vectorizer_path = os.path.join("models", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

def traducir_a_ingles(texto):
    return translator(texto)[0]["translation_text"]

def predict_sentiment(text):
    """Recibe texto plano y devuelve el sentimiento como texto: 'Positive' o 'Negative'."""
    text_en = traducir_a_ingles(text)
    x_vector = vectorizer.transform([text_en])
    prediction = model.predict(x_vector)[0]

    return prediction

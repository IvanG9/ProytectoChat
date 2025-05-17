from flask import Blueprint, request, jsonify
from services.generate_service import generate_text
from services.sentiment_service import predict_sentiment
from services.gpt2_news_service import generate_news_text

generate_bp = Blueprint("generate", __name__)

@generate_bp.route("/generate", methods=["POST"])
def generate():
    """Endpoint para generar texto con el modelo CharRNN."""
    data = request.get_json()
    text = data.get("text", "")
    model = data.get("model", "")
    max_length = data.get("max_length", 100)
    temperature = data.get("temperature", 1.0)
    print(model)
    if model == "RNN":
        generated_text = generate_text(text, max_length, temperature)
    elif model == "ST":
            sentiment = predict_sentiment(text)
            return jsonify({"sentiment": sentiment})
    elif model == "LLM":
        generated_text = generate_news_text(text, max_length, temperature)
        print("🧠 Texto generado:", generated_text)

    return jsonify({"generated_text": generated_text})
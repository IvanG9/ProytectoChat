from flask import Flask
from flask_cors import CORS
from controllers.generate_controller import generate_bp

app = Flask(__name__)
CORS(app)

# Registrar controladores
app.register_blueprint(generate_bp)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
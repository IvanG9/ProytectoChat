import torch
from models.char_rnn import CharRNN
from models.tokenizer import Tokenizer

# Inicializar Tokenizer y Modelo
tokenizer = Tokenizer()
model = CharRNN(input_size=tokenizer.n_characters)
model.load_state_dict(torch.load("./models/charRNN_model2.pth", map_location=torch.device('cpu')))
model.eval()

def generate_text(prompt, max_length=100, temperature=1.0):
    """Genera texto basado en un prompt usando el modelo RNN."""
    for _ in range(max_length):
        X_encoded = tokenizer.text_to_seq(prompt[-100:])
        X_tensor = torch.tensor(X_encoded).unsqueeze(0)
        with torch.no_grad():
            y_pred = model(X_tensor)
            y_pred = y_pred.view(-1).div(temperature).exp()
            top_i = torch.multinomial(y_pred, 1)[0].item()
            predicted_char = tokenizer.all_characters[top_i]
            prompt += predicted_char
    return prompt
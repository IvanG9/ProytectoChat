f = open("./generateModel/textos_limpios.csv", "r", encoding='utf-8')
text = f.read()
text[:300], len(text)

import string

all_characters = string.printable + "ñÑáÁéÉíÍóÓúÚ¿¡"
all_characters

class Tokenizer():

  def __init__(self):
    self.all_characters = all_characters
    self.n_characters = len(self.all_characters)

  def text_to_seq(self, string):
    seq = []
    for c in range(len(string)):
        try:
            seq.append(self.all_characters.index(string[c]))
        except:
            continue
    return seq

  def seq_to_text(self, seq):
    text = ''
    for c in range(len(seq)):
        text += self.all_characters[seq[c]]
    return text

tokenizer = Tokenizer()
tokenizer.n_characters

tokenizer.text_to_seq('señor, ¿qué tal?')
tokenizer.seq_to_text([28, 14, 100, 24, 27, 73, 94, 112, 26, 30, 104, 94, 29, 10, 21, 82])
text_encoded = tokenizer.text_to_seq(text)

train_size = len(text_encoded) * 80 // 100
train = text_encoded[:train_size]
test = text_encoded[train_size:]

len(train), len(test)

import random

def windows(text, window_size = 100):
    start_index = 0
    end_index = len(text) - window_size
    text_windows = []
    while start_index < end_index:
      text_windows.append(text[start_index:start_index+window_size+1])
      start_index += 1
    return text_windows

text_encoded_windows = windows(text_encoded)

print(tokenizer.seq_to_text((text_encoded_windows[0])))
print()
print(tokenizer.seq_to_text((text_encoded_windows[1])))
print()
print(tokenizer.seq_to_text((text_encoded_windows[2])))

import torch

class CharRNNDataset(torch.utils.data.Dataset):
  def __init__(self, text_encoded_windows, train=True):
    self.text = text_encoded_windows
    self.train = train

  def __len__(self):
    return len(self.text)

  def __getitem__(self, ix):
    if self.train:
      return torch.tensor(self.text[ix][:-1]), torch.tensor(self.text[ix][-1])
    return torch.tensor(self.text[ix])
  
train_text_encoded_windows = windows(train)
test_text_encoded_windows = windows(test)

dataset = {
    'train': CharRNNDataset(train_text_encoded_windows),
    'val': CharRNNDataset(test_text_encoded_windows)
}

dataloader = {
    'train': torch.utils.data.DataLoader(dataset['train'], batch_size=512, shuffle=True, pin_memory=True),
    'val': torch.utils.data.DataLoader(dataset['val'], batch_size=2048, shuffle=False, pin_memory=True),
}

len(dataset['train']), len(dataset['val'])

input, output = dataset['train'][0]
tokenizer.seq_to_text(input)

tokenizer.seq_to_text([output])

class CharRNN(torch.nn.Module):
  def __init__(self, input_size, embedding_size=128, hidden_size=256, num_layers=2, dropout=0.2):
    super().__init__()
    self.encoder = torch.nn.Embedding(input_size, embedding_size)
    self.rnn = torch.nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
    self.fc = torch.nn.Linear(hidden_size, input_size)

  def forward(self, x):
    x = self.encoder(x)
    x, h = self.rnn(x)
    y = self.fc(x[:,-1,:])
    return y
  
model = CharRNN(input_size=tokenizer.n_characters)
outputs = model(torch.randint(0, tokenizer.n_characters, (64, 50)))
outputs.shape


from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def fit(model, dataloader, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = []
        bar = tqdm(dataloader['train'])
        for batch in bar:
            X, y = batch
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            bar.set_description(f"loss {np.mean(train_loss):.5f}")
        bar = tqdm(dataloader['val'])
        val_loss = []
        model.eval()
        with torch.no_grad():
            for batch in bar:
                X, y = batch
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = criterion(y_hat, y)
                val_loss.append(loss.item())
                bar.set_description(f"val_loss {np.mean(val_loss):.5f}")
        print(f"Epoch {epoch}/{epochs} loss {np.mean(train_loss):.5f} val_loss {np.mean(val_loss):.5f}")

def predict(model, X):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X).to(device)
        pred = model(X.unsqueeze(0))
        return pred

model = CharRNN(input_size=tokenizer.n_characters)
fit(model, dataloader, epochs=100)

X_new = "En un lugar de la mancha, "
X_new_encoded = tokenizer.text_to_seq(X_new)
y_pred = predict(model, X_new_encoded)
y_pred = torch.argmax(y_pred, axis=1)[0].item()
tokenizer.seq_to_text([y_pred])

for i in range(100):
  X_new_encoded = tokenizer.text_to_seq(X_new[-100:])
  y_pred = predict(model, X_new_encoded)
  y_pred = torch.argmax(y_pred, axis=1)[0].item()
  X_new += tokenizer.seq_to_text([y_pred])

X_new

temp=1
for i in range(1000):
  X_new_encoded = tokenizer.text_to_seq(X_new[-100:])
  y_pred = predict(model, X_new_encoded)
  y_pred = y_pred.view(-1).div(temp).exp()
  top_i = torch.multinomial(y_pred, 1)[0]
  predicted_char = tokenizer.all_characters[top_i]
  X_new += predicted_char

print(X_new)

# Guardar solo los pesos del modelo (state_dict)
torch.save(model.state_dict(), "./models/charRNN_model2.pth")

print("Modelo guardado con éxito!")
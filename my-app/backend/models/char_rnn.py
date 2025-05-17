import torch

class CharRNN(torch.nn.Module):
    def __init__(self, input_size, embedding_size=128, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.encoder(x)
        x, _ = self.rnn(x)
        y = self.fc(x[:, -1, :])
        return y
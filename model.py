import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TimeSeriesMultiLabelDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class CNNBiLSTMMultiLabelClassifier(nn.Module):
    def __init__(self, num_labels=5, lstm_hidden=64, dropout=0.3):
        super().__init__()

        self.cnn_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.bilstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        combined_dim = 128 + (2 * lstm_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_labels)
        )

    def forward(self, x):
        # x shape: (batch, length)

        x_cnn = x.unsqueeze(1)
        cnn_features = self.cnn_branch(x_cnn).squeeze(-1)

        x_lstm = x.unsqueeze(-1)
        _, (h_n, _) = self.bilstm(x_lstm)

        forward_last = h_n[-2]
        backward_last = h_n[-1]
        lstm_features = torch.cat([forward_last, backward_last], dim=1)

        combined = torch.cat([cnn_features, lstm_features], dim=1)

        logits = self.classifier(combined)
        return logits
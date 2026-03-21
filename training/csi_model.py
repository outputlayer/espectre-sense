#!/usr/bin/env python3
"""CNN-LSTM model for WiFi CSI room presence classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Simple attention mechanism over temporal dimension."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden)
        scores = self.attn(lstm_out).squeeze(-1)  # (batch, seq_len)
        weights = F.softmax(scores, dim=1)         # (batch, seq_len)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden)
        return context, weights


class CSINet(nn.Module):
    """CNN-LSTM with attention for WiFi CSI classification.

    Input: (batch, seq_len=100, features=168)
    Output: (batch, num_classes=4)
    """
    def __init__(self, input_features=168, num_classes=4, 
                 cnn_channels=[64, 128], lstm_hidden=64, dropout=0.3):
        super().__init__()

        # 1D CNN over temporal dimension
        # Input: (batch, features, seq_len) after transpose
        layers = []
        in_ch = input_features
        for out_ch in cnn_channels:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ])
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0,
        )

        # Attention
        self.attention = Attention(lstm_hidden * 2)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        # CNN expects (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.cnn(x)        # (batch, cnn_out, seq_len//4)
        x = x.transpose(1, 2)  # (batch, seq_len//4, cnn_out)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len//4, hidden*2)

        # Attention pooling
        context, attn_weights = self.attention(lstm_out)  # (batch, hidden*2)

        # Classify
        logits = self.classifier(context)  # (batch, num_classes)
        return logits


class CSINetLight(nn.Module):
    """Lighter version for faster CPU training.
    
    Uses 1D convolutions + global average pooling, no LSTM.
    """
    def __init__(self, input_features=168, num_classes=4, dropout=0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(input_features, 128, kernel_size=7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            # Block 2
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            # Block 3
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2)       # (batch, features, seq_len)
        x = self.features(x)        # (batch, 128, 1)
        x = x.squeeze(-1)           # (batch, 128)
        return self.classifier(x)   # (batch, num_classes)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test both models
    batch = torch.randn(4, 100, 168)
    
    model1 = CSINet()
    out1 = model1(batch)
    print(f"CSINet:      {out1.shape}, params: {count_params(model1):,}")

    model2 = CSINetLight()
    out2 = model2(batch)
    print(f"CSINetLight: {out2.shape}, params: {count_params(model2):,}")

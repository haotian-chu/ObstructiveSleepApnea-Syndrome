import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, )
        return x * y.expand_as(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() *
                    -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return x

class SENet_Transformer(nn.Module):
    def __init__(self, num_classes=3):
        super(SENet_Transformer, self).__init__()

        self.conv1 = nn.Conv1d(1, 128, kernel_size=20, stride=3, padding=9)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2, stride=3)

        self.conv2 = nn.Conv1d(128, 32, kernel_size=7, stride=1, padding=3)
        self.se1 = SELayer(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2, stride=2)
        self.downsample2 = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=1),
            nn.BatchNorm1d(32)
        )

        self.conv3 = nn.Conv1d(32, 32, kernel_size=10, stride=1, padding=5)
        self.se2 = SELayer(32)
        self.pool3 = nn.MaxPool1d(2, stride=2)

        self.d_model = 32  # Embedding dimension
        self.pos_encoder = PositionalEncoding(d_model=self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(self.d_model, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, num_classes)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = F.relu(out1)
        out1 = self.bn1(out1)
        out1 = self.pool1(out1)
        out1 = F.relu(out1)

        identity2 = self.downsample2(out1)
        out2 = self.conv2(out1)
        out2 = F.relu(out2)
        out2 = self.se1(out2)
        out2 = self.bn2(out2)
        out2 += identity2
        out2 = self.pool2(out2)
        out2 = F.relu(out2)

        out3 = self.conv3(out2)
        out3 = F.relu(out3)
        out3 = self.se2(out3)
        out3 = self.pool3(out3)
        out3 = F.relu(out3)

        x = out3.permute(0, 2, 1)  # Shape: (batch_size, seq_len, channels)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)

        x = x.permute(1, 0, 2)  # Shape: (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)

        x = x.mean(dim=0)  # Aggregate over the sequence dimension

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



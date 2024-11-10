import torch.nn as nn
import torch
import torch.nn.functional as F


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


class SENet_LSTM(nn.Module):
    def __init__(self, num_classes=3):
        super(SENet_LSTM, self).__init__()

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



        self.lstm = nn.LSTM(input_size=32, hidden_size=10, batch_first=True)


        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(10, 20)
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


        x = out3.permute(0, 2, 1)  # [batch_size, seq_length, channels]
        x, _ = self.lstm(x)
        x = x[:, -1, :]

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


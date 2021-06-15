import torch
import torch.nn as nn
import torch.nn.functional as F


class RCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, num_classes=2):
        super(RCNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.conv1 = nn.Conv2d(3, 64, 10, 1, padding=3, bias=True)
        self.pool1 = nn.MaxPool2d((2*2, 3))
        self.conv2 = nn.Conv2d(64, self.input_size, 5, 1, bias=True)
        self.pool2 = nn.MaxPool2d((2*2, 3))

        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, batch_first=True)

        self.fc1 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        h0 = torch.zeros(self.num_layers, 276, self.hidden_size).to(device)

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(-1, x.size(0), self.hidden_size)
        x = x.permute(0, 2, 1)

        out, _ = self.gru(x, h0)
        out = F.relu(out)

        out = out[:, -1, :]

        out = self.fc1(out)

        return out






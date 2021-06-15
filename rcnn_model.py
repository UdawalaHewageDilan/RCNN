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

        self.conv1 = nn.Conv2d(3, 4*self.input_size, 5)
        self.pool1 = nn.MaxPool2d((2*2, 3))
        self.conv2 = nn.Conv2d(4*self.input_size, 16*self.input_size, 5, padding=(4, 4))
        self.pool2 = nn.MaxPool2d((2*2, 3))
        #self.batch_norm = nn.BatchNorm2d(16*self.input_size, 1e-5)

        self.gru = nn.GRU(self.input_size, 64, batch_first=True)

        self.fc1 = nn.Linear(64, self.num_classes)

    def forward(self, x):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #
        # h0 = torch.zeros(self.num_layers, 32, self.hidden_size).to(device)

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        #x = self.batch_norm(x)

        x = x.view(32, 8, -1)
        x = x.permute(0, 2, 1)

        out, _ = self.gru(x)
        out = F.relu(out)

        #out = out[:, -1, :]

        out = self.fc1(out[:, -1])

        return F.log_softmax(out.type(torch.FloatTensor), dim=-1)






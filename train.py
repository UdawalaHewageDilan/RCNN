import torch
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.optim import lr_scheduler, Adam, SGD
from rcnn_model import RCNN
import data_handler as dh


def train(epochs, model, criterion, lr_scheduler, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    model = model.to(device)

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dh.train_loader):
            images = images.to(device)
            labels = labels.to(device)

            labels = labels.type(torch.FloatTensor).view(-1)
            print(labels.shape)
            print(labels)

            output = model(images)
            print(output)
            print(output.shape)
            loss = criterion(labels, output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}], Loss: {loss.item():.4f}')
        lr_scheduler.step()


epochs = 4

input_size = 8
hidden_size = 64
num_layers = 1
num_classes = 2

model = RCNN(input_size, hidden_size, num_layers, num_classes)

criterion = NLLLoss()

lr = 0.01
optimizer = Adam(model.parameters(), lr)

lr_scheduler = lr_scheduler.StepLR(optimizer, 1)

train(epochs, model, criterion, lr_scheduler, optimizer)










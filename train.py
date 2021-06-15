import torch
from torch.nn import CrossEntropyLoss, NLLLoss
from torch.optim import lr_scheduler, Adam, SGD
from rcnn_model import RCNN
import data_handler as dh
import matplotlib.pyplot as plt

train_losses = []
val_losses = []


def train(epochs, model, criterion, lr_scheduler, optimizer, device):

    best_val = 2
    model = model.to(device)
    #model.train()
    tr_loss = 0
    criterion = criterion.to(device)
    # lr_scheduler = lr_scheduler.to(device)
    # optimizer = optimizer.to(device)

    for epoch in range(epochs):

        print(epoch, lr)
        for i, (images, labels) in enumerate(dh.train_loader):
            model = model.to(device)
            images = images.to(device)
            labels = labels.to(device)

            labels = labels
            # print(labels.shape)
            # print(labels)

            output = model(images)
            # print(output)
            # print(output.shape)

            loss_train = criterion(output.to(device), labels)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            tr_loss += loss_train.item()
            if i % 10 == 0:
                print(loss_train)

        print(f'Epoch [{epoch + 1}/{epochs}] Loss: {loss_train.item():.4f}')
        lr_scheduler.step()
        train_losses.append(tr_loss / images.shape[0])

        val_loss = 0
        with torch.no_grad():
            for x, y in dh.test_loader:
                output = model(x.to(device))
                # output = torch.reshape(output,(output.shape[0],))
                # print(output.shape)
                loss_val = criterion(output, y.to(device))
                val_loss += loss_val.item()

        val_loss = val_loss / x.shape[0]
        val_losses.append(val_loss)
        print('Epoch : ', epoch, "\t Train loss: ", tr_loss / images.shape[0], "\t Validation loss: ", val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model, "./models/Best_model2.pth")

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 10

input_size = 8
hidden_size = 64
num_layers = 1
num_classes = 2

model = RCNN(input_size, hidden_size, num_layers, num_classes)

criterion = NLLLoss()

lr = 0.01
optimizer = Adam(model.parameters(), lr)

lr_scheduler = lr_scheduler.StepLR(optimizer, 1)

train(epochs, model, criterion, lr_scheduler, optimizer, device)










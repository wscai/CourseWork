# %% Import and load data
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import utils.mnist_reader as mnist_reader
import torch.optim as optim
import time
import random
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from helper_functions import parameter_count
def augmentation(data):
    # randomly flip the image
    if random.randint(0, 1) == 1:
        ans1 = np.fliplr(data[:])
    else:
        ans1 = data[:]
    # pad the image with 4 pixels each side
    ans1 = np.pad(ans1, (4, 4), 'constant', constant_values=0)
    # randomly crop the image
    randy = random.randint(0, 8)
    randx = random.randint(0, 8)
    ans1 = ans1[randy:randy + 28, randx:randx + 28]
    # return a tensor with correct type
    return torch.from_numpy(ans1).float()


class Data(torch.utils.data.Dataset):
    # 0 for train, 1 for test, 2 for validation
    def __init__(self, train=0):
        # load data
        if train == 0:
            self.X, self.Y = mnist_reader.load_mnist('data/fashion', kind='train')
            self.X = self.X[:-1000]
            self.Y = self.Y[:-1000]
        elif train == 1:
            self.X, self.Y = mnist_reader.load_mnist('data/fashion', kind='t10k')
        else:
            self.X, self.Y = mnist_reader.load_mnist('data/fashion', kind='train')
            self.X = self.X[-1000:]
            self.Y = self.Y[:][-1000:]
        # normalize data to from [0,255] to [-1,1]
        self.X = self.X / 255 * 2 - 1
        # reshape data
        self.X = self.X.reshape(self.X.shape[0], 28, 28)
        # convert labels to tensor
        self.Y = torch.Tensor(self.Y.tolist()).long()

    def __getitem__(self, index):
        # return augmented data and label
        return augmentation(self.X[index]), self.Y[index]

    def __len__(self):
        return len(self.X)


# load data
train_set = Data()
test_set = Data(train=1)
valid_set = Data(train=2)

# Tensorboard
writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# CNN Model and hyperparameters
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # convolutional layers
        self.seq = nn.Sequential(
            # 5×5 Convolutional Layer with 32 filters, stride 1 and padding 2
            nn.Conv2d(1, 32, 5, 1, 2),
            # ReLU activation layer
            nn.ReLU(),
            # 2×2 Max Pooling Layer with stride 2
            nn.MaxPool2d(2, 2),
            # 3x3 Convolutional Layer with 64 filters, stride 1 and padding 1
            nn.Conv2d(32, 64, 3, 1, 1),
            # ReLU activation layer
            nn.ReLU(),
            # 2×2 Max Pooling Layer with stride 2
            nn.MaxPool2d(2, 2),
            # Flatten the output
            nn.Flatten()
        )
        # fully connected layers
        self.fc = nn.Sequential(
            # Fully-connected layer with 1024 output units
            nn.Linear(3136, 1024),
            # ReLU activation layer
            nn.ReLU(),
            # Fully-connected layer with 10 output units
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        # forward pass to conv layers
        x = self.seq(x)
        # forward pass to fc layers
        x = self.fc(x)
        return x
parameter_count(Model())

# %% Training
# hyperparameters
lr = 0.001
betas = (0.9, 0.999)
batch_size = 200
epochs = 100

# initialize model, loss function, optimizer, dataloader
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')

# initialize model
model = Model().to(device)
# set up cross entropy loss
loss = nn.CrossEntropyLoss().to(device)
# set up Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas)
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
# record training time
t = time.time()
# training loop
for epoch in range(epochs):
    running_loss = 0.0
    running_correct = 0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs = inputs.reshape(inputs.shape[0], 1, 28, 28).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_size = loss(outputs, labels)
        loss_size.backward()
        optimizer.step()
        running_loss += loss_size.item()
        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
    with torch.no_grad():
        # calculate accuracy on validation set
        correct = 0
        val_loss = 0
        for data in valid_dataloader:
            inputs, labels = data
            inputs = inputs.reshape(inputs.shape[0], 1, 28, 28)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            val_loss += loss(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        print(f'Epoch {epoch + 1}, training loss {running_loss / len(train_set)}, training accuracy {running_correct/len(train_set)}, '
              f'validation loss {val_loss/len(valid_set)}, validation accuracy {correct / len(valid_set)}.')
        # add to tensorboard
        writer.add_scalar('Loss/train', running_loss / len(train_set), epoch)
        writer.add_scalar('Loss/validation', val_loss / len(valid_set), epoch)
        writer.add_scalar('Accuracy/train', running_correct / len(train_set), epoch)
        writer.add_scalar('Accuracy/validation', correct / len(valid_set), epoch)
# record training time
t = time.time() - t
print(f'Finished training in {t} seconds, {t/epochs} seconds per epoch')
# test accuracy
with torch.no_grad():
    correct = 0
    test_loss = 0
    for data in test_dataloader:
        inputs, labels = data
        inputs = inputs.reshape(inputs.shape[0], 1, 28, 28).to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        test_loss += loss(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
    print(f'Test loss {test_loss/len(test_set)}, test accuracy: {correct / len(test_set)}')
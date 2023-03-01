import torch
import torch.nn as nn
import pickle
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random
import pathlib
import ssl
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
from statistics import mean
# Hyper Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 100
batch_size = 128
learning_rate = 0.001
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615)),
    ])
transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615)),
])
transform2 = transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615)),
])

transform3 = transforms.Compose([transforms.AutoAugment(transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615)),
])
# transform4 = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomVerticalFlip(p=1.0),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465),
#                          (0.247, 0.2434, 0.2615)),
# ])
train_dataset_1 = dsets.CIFAR10(root='./data/',
                               train=True,
                               transform=transform1,
                               download=True)
train_dataset_2 = dsets.CIFAR10(root='./data/',
                               train=True,
                                 transform=transform2,
                                 download=True)
train_dataset_3=dsets.CIFAR10(root='./data/',
                              train=True,
                              transform=transform3,
                              download=True)
# train_dataset_4=dsets.CIFAR10(root='./data/',
#                                 train=True,
#                                 transform=transform4
#                                 ,download=True)
train_dataset = torch.utils.data.ConcatDataset([train_dataset_1,train_dataset_2])
train_dataset = torch.utils.data.ConcatDataset([train_dataset,train_dataset_3])
test_dataset = dsets.CIFAR10(root='./data/',
                              train=False,
                              transform=transform1,
                              download=True)
# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        # nn.Dropout(0.5),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        # nn.Dropout(0.5),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 32, kernel_size=3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 16, kernel_size=3, padding=1),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(16, 16, kernel_size=3, padding=1),
                        nn.BatchNorm2d(16),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(16, 10, kernel_size=1, padding=1),
                        nn.BatchNorm2d(10),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        nn.Conv2d(10, 10, kernel_size=1, padding=1),
                        nn.MaxPool2d(kernel_size=2),


        )
        # self.dropout = nn.Dropout(p=0.25)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer(x)
        # out = out.view(-1, 10)
        #Squeeze the output to get the required output
        out = out.squeeze(2)
        out = out.squeeze(2)
        out = self.logsoftmax(out)
        return out



cnn = CNN()

if torch.cuda.is_available():
    cnn = cnn.cuda()

# convert all the weights tensors to cuda()
# Loss and Optimizer
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
print('number of parameters: ', sum(param.numel() for param in cnn.parameters()))
train_losses = []
test_losses = []
train_acc = []
test_acc = []
for epoch in range(num_epochs):
    correct_train=0
    total_train=0
    train_loss_epoch=[]
    train_acc_epoch=[]
    test_acc_epoch=[]
    test_loss_epoch=[]
    for i, (images, labels) in enumerate(train_loader):
        correct_train= 0
        total_train = 0
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
            # Forward + Backward + Optimize
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train+= (predicted == labels).sum().item()
        train_acc_epoch.append(100 * correct_train / total_train)
        train_loss_epoch.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print('Train: Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1,
                     len(train_dataset) // batch_size, loss.data))
    correct = 0
    total = 0
    with torch.no_grad():
        for images2, labels2 in test_loader:
            if torch.cuda.is_available():
                images2 = images2.to(device)
                labels2 = labels2.to(device)
            outputs2 = cnn(images2)
            _, predicted = torch.max(outputs2.data, 1)
            total += labels2.size(0)
            correct += (predicted == labels2).sum()
            loss = criterion(outputs2, labels2)
            test_loss_epoch.append(loss.item())

    train_acc.append(np.mean(train_acc_epoch))
    train_losses.append(np.mean(train_loss_epoch))
    test_acc.append(100 * correct / total)
    test_losses.append(np.mean(test_loss_epoch))
    print("Test Accuracy of the model on the 10000 test images: ",(100 * correct / total))
    print("Train Accuracy of the model for this epoch: ",(100 * correct_train / total_train))
    if 100 * correct / total>81:
        break

with open("q1_cnn2.pkl", "wb") as f:
        pickle.dump(cnn, f)
#Plot the loss and accuracy curves for training and test
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(frameon=False)
plt.show()
plt.subplot(1,2,2)
#Change accuracy to error,error =1-accuracy
train_error = [1 - x / 100 for x in train_acc]
test_error = [1 - x / 100 for x in test_acc]
plt.plot(train_error, label='Training error')
plt.plot(test_error, label='Test error')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.legend(frameon=False)
plt.show()

     

        
        


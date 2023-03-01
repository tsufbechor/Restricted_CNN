import numpy as np
import time
import math
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torchvision.transforms import ToTensor
import pickle
batch_size = 128
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
            # Squeeze the output to get the required output
            out = out.squeeze(2)
            out = out.squeeze(2)
            out = self.logsoftmax(out)
            return out
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
def evaluate_hw1():
    test = enumerate(test_loader)
    loss_function = nn.NLLLoss()
    test_loss=[]
    test_accuracy=[]
    model = pickle.load(open("model_q1.pkl", 'rb'))
    total=0
    correct=0
    with torch.no_grad():
        for images2, labels2 in test_loader:
            if torch.cuda.is_available():
                images2 = images2.to(device)
                labels2 = labels2.to(device)
            outputs2 = model(images2)
            _, predicted = torch.max(outputs2.data, 1)
            total += labels2.size(0)
            correct += (predicted == labels2).sum()
    #Print error of the model
    print('Test Error of the model on the 10000 test images: {} %'.format(100-(100 * correct / total)))




if __name__ == '__main__':
    evaluate_hw1()
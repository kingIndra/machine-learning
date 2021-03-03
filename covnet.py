import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam


class convo_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(convo_layer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self, inp):
        out = self.conv(inp)
        out = self.batchnorm(out)
        out = self.relu(out)

        return out


class ConvoNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvoNet, self).__init__()

        self.conv1 = convo_layer(in_channels=3, out_channels=32)
        self.conv2 = convo_layer(in_channels=32, out_channels=32)
        self.conv3 = convo_layer(in_channels=32, out_channels=32)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = convo_layer(in_channels=32, out_channels=64)
        self.conv5 = convo_layer(in_channels=64, out_channels=64)
        self.conv6 = convo_layer(in_channels=64, out_channels=64)
        self.conv7 = convo_layer(in_channels=64, out_channels=64)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv8 = convo_layer(in_channels=64, out_channels=128)
        self.conv9 = convo_layer(in_channels=128, out_channels=128)
        self.conv10 = convo_layer(in_channels=128, out_channels=128)
        self.conv11 = convo_layer(in_channels=128, out_channels=128)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv12 = convo_layer(in_channels=128, out_channels=128)
        self.conv13 = convo_layer(in_channels=128, out_channels=128)
        self.conv14 = convo_layer(in_channels=128, out_channels=128)

        self.avgpool = nn.AvgPool2d(kernel_size=4)

        self.network = nn.Sequential(self.conv1, self.conv2, self.conv3, self.pool1, self.conv4, self.conv5, self.conv6, self.conv7,
                                     self.pool2, self.conv8, self.conv9, self.conv10, self.conv11, self.pool3, self.conv12, self.conv13, self.conv14, self.avgpool)

        self.fc = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, inp):
        out = self.network(inp)
        out = out.view(-1, 128)
        out = self.fc(out)
        return out


train_transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = CIFAR10(root="./data", train=True,
                    transform=train_transformations, download=False)

train_loader = DataLoader(train_set, batch_size=32,
                          shuffle=True, num_workers=4)

test_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_set = CIFAR10(root="./data", train=False,
                   transform=test_transformations, download=False)

test_loader = DataLoader(test_set, batch_size=32,
                         shuffle=False, num_workers=4)

# Initiating the Model;
model = ConvoNet(num_classes=10)
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
loss_fn = nn.CrossEntropyLoss()


def adjust_lr(epochs):
    lr = 0.0001
    if epochs >= 30:
        x = int(epochs/30)
        y = pow(10, x)
        lr = lr/y
    for param_groups in optimizer.param_groups:
        param_groups["lr"] = lr


def save_model(epochs):
    torch.save(model.state_dict(), f'CIFAR10_{epochs}.model')
    print('CheckPoint Saved')


def test_model():
    model.eval()
    test_accuracy = 0.0

    for i, (images, labels) in enumerate(test_loader):
        output = model(images)
        _, pred = torch.max(output.data, 1)

        test_accuracy += torch.sum(pred == labels.data)

    test_accuracy = test_accuracy / 10000
    return test_accuracy


def train_model(EPOCHS):
    max_accuracy = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_acc = 0.0
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(images)

            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().data[0] * images.size(0)
            _, pred = torch.max(output.data, 1)

            train_acc += torch.sum(pred == labels.data)

        adjust_lr(epoch)
        train_acc /= 50000
        train_loss /= 50000

        test_acc = test_model()
        if test_acc > max_accuracy:
            save_model(epoch)
            max_accuracy = test_acc

        print("Epoch {}, Train Accuracy: {} , TrainLoss: {} , Test Accuracy: {}".format(
            epoch, train_acc, train_loss, test_acc))


if __name__ == '__main__':
    # train_model(200)
    print('Building on Jenkins for Now')

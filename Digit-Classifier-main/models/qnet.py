import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from qintegration.qmodule import ExampleQLayer

class QSNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(784, 84)
        # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 3)
        self.q1 = ExampleQLayer(3)
        self.fc5 = nn.Linear(3, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x)) * 0.8
        x = F.relu(self.q1(x))
        x = self.fc5(x)
        # x = F.softmax(x)
        return x

class QCNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 3)
        self.q1 = ExampleQLayer(3)
        self.fc5 = nn.Linear(3, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x)) * 0.8
        x = F.relu(self.q1(x))
        x = self.fc5(x)
        # x = F.softmax(x)
        return x
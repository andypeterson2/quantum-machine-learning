import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.switchable import Switchable

class Polynomial(nn.Module):
    """
    The nn.Module that acts as a polynomial layer. 
    A polynomial layer takes input vector x and outputs vector y = [exp(x, w_1), exp(x, w_2), ...],
    where each w_i is a vector of weights and exp(u, v) = e ** v_0 * (1 + |u_1|) ** v_1 * (1 + |u_2|) ** v_2 * ....

    Attributes
    ----------
    input_dim : int
        The dimension of input vector x
    output_dim : int
        The dimention of output vector y
    fc : nn.Linear
        The weight matrix storing all w_i
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Default constructor for Polynomial
    
        Parameters
        ----------
        input_dim : int
            The dimension of input vector x
        output_dim : int
            The dimention of output vector y
        """

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.log(torch.abs(x) + 1)
        x = self.fc(x)
        x = torch.exp(x)
        return x

class PolynomialNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.poly_dim1 = (120, 84)
        self.poly1 = Switchable(
            [
                Polynomial(*self.poly_dim1), 
                nn.Linear(*self.poly_dim1)
            ]
        )
        self.fc3 = nn.Linear(84, 32)
        self.poly_dim2 = (32, 16)
        self.poly2 = Switchable(
            [
                Polynomial(*self.poly_dim2), 
                nn.Linear(*self.poly_dim2)
            ]
        )
        self.fc5 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.poly1(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.poly2(x))
        x = self.fc5(x)
        return x
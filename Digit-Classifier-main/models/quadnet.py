import torch
import torch.nn as nn
import torch.nn.functional as F

from models.switchable import Switchable

class Quadratic(nn.Module):
    """
    The nn.Module that acts as a quadratic layer. 
    A quadratic layer takes input vector x and outputs vector 
    y = concat([
        W_01 * x_1, W_02 * x_2, ..., 

        W_11 * x_1 * x_1, W_12 * x_1 * x_2, ..., 

        W_21 * x_2 * x_1, W_22 * x_2 * x_2, ...
    ]).

    Attributes
    ----------
    input_dim : int
        The dimension of input vector x
    output_dim : int
        The dimention of output vector y
    fc : nn.Linear
        The weight matrix storing all W_ij

    Methods
    -------
    expand(x: torch.Tensor) -> torch.Tensor:
        returns vector z, where y = Wz
    """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Default constructor for Quadratic
    
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
        self.fc = nn.Linear(input_dim * (input_dim + 1), output_dim)

    def expand(x: torch.Tensor) -> torch.Tensor:
        """
        Expands the input vector x into vector z = (x.T @ x | x).flatten().
        z is the vector that is multiplied by weight matrix W to get y.
        y = Wz.
    
        Parameters
        ----------
        x : torch.Tensor
            The input vector
    
        Returns
        -------
        z : torch.Tensor
            The quadratic expandsion of x, such that y = Wz.
        """

        x = x.view(x.shape[0], 1, *x.shape[1:]) # insert a dim after batch for transposing
        xt = torch.transpose(x, 1, 2) # transpose per sample
        x = torch.concat((x, torch.ones((x.shape[0], 1, 1))), dim=-1) # append mul id to retain original linear relationships

        z = torch.flatten(xt @ x, start_dim=1, end_dim=-1) # cross every pair of inputs to get all quadratic relationships
        return z

    def forward(self, x):
        x = Quadratic.expand(x)
        x = self.fc(x)
        return x

class QuadraticNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(120, 32)
        self.quad_dim = (32, 16)
        self.quad = Switchable(
            [
                Quadratic(*self.quad_dim), 
                nn.Sequential(
                    nn.Linear(self.quad_dim[0], self.quad_dim[0] * (self.quad_dim[0] + 1)),
                    nn.Linear(self.quad_dim[0] * (self.quad_dim[0] + 1), self.quad_dim[1])
                )
            ]
        )
        self.fc5 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.quad(x))
        x = self.fc5(x)
        return x
from typing import Any, Optional

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from train.eval import get_acc
from train.config import device

class Validate(ABC):
    """
    A validation class used to evaluate the model on validation data during training, 
    and report the results and any performed validations.

    Attributes
    ----------
    history: list[Any]
        The history of all validations performed by this object
    verbose: bool
        Whether to print the result of each validation

    Methods
    -------
    show_history(history: Optional[list[Any]] = None) -> list[Any]
        Display the recorded history of this Validation object, or another history if given, and return the displayed history
    print(*args)
        Same as python print function, but only print if this object is verbose
    """
    def __init__(self, verbose: bool = True):
        """
        Default constructor for Validate
    
        Parameters
        ----------
        history: list[Any]
            The history of all validations performed by this object
        verbose: bool
            Whether to print the result of each validation
        """
        self.history = []
        self.verbose = verbose

    # return acc, record
    @abstractmethod
    def __call__(self, epoch: int, i: int, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor, train_loss: torch.Tensor) -> float | None:
        pass
    
    @abstractmethod
    def show_history(self, history: Optional[list[Any]] = None) -> list[Any]:
        """
        Display the recorded history of this Validation object, or another history if given, and return the displayed history

        Parameters
        ----------
        history: list[Any] | None
            The history to display instead of this Validation object's recorded history

        Returns
        -------
        history : list[Any]
            The history that was displayed by this method
        """

        history = history or self.history
        return history

    def print(self, *args):
        """
        Same as python default print function, but only print if this object is verbose

        Parameters
        ----------
        *args
            The arguments to pass to the python default print function
        """

        if not self.verbose: return
        print(*args)

class NoValidate(Validate):

    def __init__(self):
        """
        Default constructor for NoValidate, invokes Validate constructor with verbose=False.
        """
        super().__init__(verbose=False)

    def __call__(self, epoch, i, model, inputs, labels, train_loss) -> float | None:
        pass

    def show_history(self, history=None) -> None:
        pass

class ExampleValidate(Validate):
    """
    An example Validate class that validates every some number batches and records training loss and validation accuracy
    """
    def __init__(self, validloader: DataLoader, val_gap: int = 10, verbose: bool = True):
        """
        Default constructor for ExampleValidate
    
        Parameters
        ----------
        valloader: Dataloader
            The validation data to use
        val_gap: int
            The number of batches in between validations
        verbose: bool
            Whether to print the result of each validation
        """
        super().__init__(verbose=verbose)
        self.validloader = validloader
        self.val_gap = val_gap

        self.running_loss = 0

    def __call__(self, epoch, i, model, inputs, labels, train_loss):
        self.running_loss += train_loss
        
        if (i+1) % self.val_gap != 0: return None

        outputs = model(inputs)
        self.print(f'[{epoch + 1}, {i+1}] loss: {self.running_loss / self.val_gap:.3f}')
        self.print(outputs[0], labels[0])

        with torch.no_grad():
            acc, errors = get_acc(model, self.validloader)
        self.print(f"Validation Accuracy: {acc:.3f}")

        self.running_loss = 0

        self.history.append([self.running_loss / self.val_gap, acc])
        # self.history.append([self.running_loss / self.val_gap, acc, outputs[0][0].item()])

        return acc
    
    def show_history(self, history=None):
        history = super().show_history(history)

        plt.plot(history)
        plt.legend(["Training loss", "Validation Accuracy"])
        # plt.legend(["Training loss", "Validation Accuracy", "Sample Output"])
        plt.show()
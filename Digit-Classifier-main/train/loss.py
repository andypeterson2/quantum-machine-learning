from typing import Callable

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from train.config import device

class Loss(ABC):
    """
    The generic loss class used for the training methods specified in train
    """
    # return loss
    @abstractmethod
    def __call__(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> torch.FloatTensor:
        pass

class TrueLoss(Loss):
    """
    A loss class for training the model on the true data

    Attributes
    ----------
    criterion: (torch.Tensor, torch.Tensor) -> torch.Tensor
        The loss function that takes in the model outputs and actual labels and returns the loss
    regularization: (nn.Module) -> torch.Tensor
        The regularization function that takes in the model and returns the regularization penalty
    """
    def __init__(
            self, 
            criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
            regularization: Callable[[nn.Module], torch.Tensor] = lambda model: torch.tensor([0])
        ):
        """
        Default constructor for TrueLoss
    
        Parameters
        ----------
        criterion: (torch.Tensor, torch.Tensor) -> torch.Tensor
            The loss function that takes in the model outputs and actual labels and returns the loss
        regularization: (nn.Module) -> torch.Tensor
            The regularization function that takes in the model and returns the regularization penalty
        """
        self.criterion = criterion
        self.regularization = regularization

    def __call__(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        outputs = model(inputs)
        return self.criterion(outputs, labels) + self.regularization(model)
    
class DistillLoss(Loss):
    """
    A loss class for distilling the model (student) with another model (teacher). 
    Returns a mix of the distillation loss and true loss.

    Attributes
    ----------
    true_loss: TrueLoss
        The TrueLoss object for the loss of the student model on the true data
    teacher: nn.Module
        The model whose knowledge to distill into the student model
    distillation_criterion: (torch.Tensor, torch.Tensor) -> torch.Tensor
        The loss function that takes in the student model outputs and teacher model predictions and returns the distillation loss
    teacher_process: (torch.Tensor) -> torch.Tensor
        Any post processing to apply to the teacher outputs to obtain the teacher predictions
    weight: float
        The proportion of the final loss that should be the distillation loss (as opposed to the true loss)
    """
    def __init__(
            self, 
            true_loss: TrueLoss, 
            teacher: nn.Module, 
            distillation_criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
            teacher_process: Callable[[torch.Tensor], torch.Tensor] = lambda x: x, 
            weight: float = 0.5
        ):
        """
        Default constructor for DistillLoss
    
        Parameters
        ----------
        true_loss: TrueLoss
            The TrueLoss object for the loss of the student model on the true data
        teacher: nn.Module
            The model whose knowledge to distill into the student model
        distillation_criterion: (torch.Tensor, torch.Tensor) -> torch.Tensor
            The loss function that takes in the student model outputs and teacher model predictions and returns the distillation loss
        teacher_process: (torch.Tensor) -> torch.Tensor
            Any post processing to apply to the teacher outputs to obtain the teacher predictions
        weight: float
            The proportion of the final loss that should be the distillation loss (as opposed to the true loss)
        """
        self.true_loss = true_loss
        self.teacher = teacher
        self.distillation_criterion = distillation_criterion
        self.teacher_process = teacher_process
        self.weight = weight

    def __call__(self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        outputs = model(inputs)

        with torch.no_grad():
            teacher_preds = self.teacher_process(self.teacher(inputs).detach())
        distillation_loss = self.distillation_criterion(outputs, teacher_preds)

        return (1 - self.weight) * self.true_loss(model, inputs, labels) + self.weight * distillation_loss
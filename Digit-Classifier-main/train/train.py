from typing import Optional, Callable

import copy
import torch
from torch import optim
import torch.nn as nn
from train.config import device
from train.loss import Loss, TrueLoss, DistillLoss
from train.validate import Validate, NoValidate, ExampleValidate
from torch.utils.data import DataLoader

def _train(
        model: nn.Module, 
        trainloader: DataLoader, 
        get_loss: Loss, 
        lr: float = 0.002, 
        epochs: int = 15, 
        patience: int = 3, 
        validate: Validate = NoValidate,
        best: Optional[tuple[float, nn.Module]] = None
    ) -> tuple[tuple[float, nn.Module], Validate]:
    """
    Train the model on the given dataloader

    Parameters
    ----------
    model : nn.Module
        The model to train
    trainloader : Dataloader
        The dataloader that contains the data for the model to be trained on
    get_loss : Loss
        The train-compatible Loss object with which to calculate the loss of the model on some data point
    lr : float
        The learning rate
    epochs : int
        The number of epochs to train the model for
    patience : int
        The number of epochs to continue training the model without seeing any improvement
    validate : Validate
        The validation object to evaluate the model on validation data during training
    best : tuple[float, nn.Module]
        The current best accuracy and the model that achieved it

    Returns
    -------
    best : tuple[float, nn.Module]
        The final best accuracy and the model that achieved it
    validate : Validate
        The validation object that evaluated the model during training and recorded its performance history
    """
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc, best_model = best or (0, copy.deepcopy(model))

    best_epoch = -1
    for epoch in range(epochs):
        
        for i, data in enumerate(trainloader):
            # Data
            inputs, labels = data
            # labels = torch.tensor([1. if labels.item() == x else 0. for x in classes]).view(1, -1)
            inputs, labels = inputs.to(device), labels.to(device)


            # Training
            optimizer.zero_grad()
            loss = get_loss(model, inputs, labels)
            loss.backward()
            optimizer.step()
            

            # Validation
            acc = validate(epoch, i, model, inputs, labels, loss)
            if not acc: continue

            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_model.load_state_dict(model.state_dict())

            if acc == 1: break

        if acc == 1: break

        if best_acc > 0.6 and epoch > best_epoch + patience: break

    print('Finished Training')
    return (best_acc, best_model), validate
    
def train(
        model: nn.Module, 
        trainloader: DataLoader, 
        validloader: DataLoader, 
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        regularization: Callable[[nn.Module], torch.Tensor] = lambda model: torch.tensor([0]),
        lr: float = 0.002, 
        epochs: int = 15, 
        patience: int = 3, 
        val_gap: int = 10, 
        verbose: bool = True,
        best: Optional[tuple[float, nn.Module]] = None
    ) -> tuple[tuple[float, nn.Module], Validate]:
    """
    Train the model on the given dataloader

    Parameters
    ----------
    model : nn.Module
        The model to train
    trainloader : Dataloader
        The dataloader that contains the data for the model to be trained on
    validloader : Dataloader
        The dataloader that contains the data for the model to be validated on
    criterion: (torch.Tensor, torch.Tensor) -> torch.Tensor
        The loss function that takes in the model outputs and actual labels and returns the loss
    regularization: (nn.Module) -> torch.Tensor
        The regularization function that takes in the model and returns the regularization penalty
    lr : float
        The learning rate
    epochs : int
        The number of epochs to train the model for
    patience : int
        The number of epochs to continue training the model without seeing any improvement
    val_gap : int
        The number of batches between each validation
    verbose : bool
        Whether to print the result of each validation
    best : tuple[float, nn.Module]
        The current best accuracy and the model that achieved it

    Returns
    -------
    best : tuple[float, nn.Module]
        The final best accuracy and the model that achieved it
    validate : Validate
        The validation object that evaluated the model during training and recorded its performance history
    """

    return _train(
        model, trainloader, TrueLoss(criterion, regularization),
        lr=lr, epochs=epochs, patience=patience,
        validate=ExampleValidate(validloader, val_gap=val_gap, verbose=verbose),
        best=best
    )

def distill(
        model: nn.Module, 
        trainloader: DataLoader, 
        validloader: DataLoader, 
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
            
        teacher: nn.Module, 
        distillation_criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
        teacher_process: Callable[[torch.Tensor], torch.Tensor] = lambda x: x, 
        weight: float=0.5,
        
        regularization: Callable[[nn.Module], torch.Tensor] = lambda model: torch.tensor([0]),
        lr=0.002, 
        epochs=15, 
        patience=3, 
        val_gap=10, 
        verbose=True,
        best=None
    ) -> tuple[tuple[float, nn.Module], Validate]:
    """
    Distill the model on the given teacher model and dataloader

    Parameters
    ----------
    model : nn.Module
        The model to train
    trainloader : Dataloader
        The dataloader that contains the data for the model to be trained on
    validloader : Dataloader
        The dataloader that contains the data for the model to be validated on
    criterion: (torch.Tensor, torch.Tensor) -> torch.Tensor
        The loss function that takes in the model outputs and actual labels and returns the loss
    teacher: nn.Module
        The model whose knowledge to distill into the student model
    distillation_criterion: (torch.Tensor, torch.Tensor) -> torch.Tensor
        The loss function that takes in the student model outputs and teacher model predictions and returns the distillation loss
    teacher_process: (torch.Tensor) -> torch.Tensor
        Any post processing to apply to the teacher outputs to obtain the teacher predictions
    weight: float
        The proportion of the final loss that should be the distillation loss (as opposed to the true loss)
    regularization: (nn.Module) -> torch.Tensor
        The regularization function that takes in the model and returns the regularization penalty
    lr : float
        The learning rate
    epochs : int
        The number of epochs to train the model for
    patience : int
        The number of epochs to continue training the model without seeing any improvement
    val_gap : int
        The number of batches between each validation
    verbose : bool
        Whether to print the result of each validation
    best : tuple[float, nn.Module]
        The current best accuracy and the model that achieved it

    Returns
    -------
    best : tuple[float, nn.Module]
        The final best accuracy and the model that achieved it
    validate : Validate
        The validation object that evaluated the model during training and recorded its performance history
    """

    return _train(
        model, trainloader, DistillLoss(TrueLoss(criterion, regularization), teacher, distillation_criterion, teacher_process=teacher_process, weight=weight),
        lr=lr, epochs=epochs, patience=patience,
        validate=ExampleValidate(validloader, val_gap=val_gap, verbose=verbose),
        best=best
    )
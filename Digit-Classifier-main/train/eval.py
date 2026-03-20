import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from train.config import device
from torch.utils.data import DataLoader
from collections import defaultdict

def get_acc(model: nn.Module, dataloader: DataLoader):
    """
    Get the accuracy and errors of the given model on the given dataloader

    Parameters
    ----------
    model : nn.Module
        The model to evaluate
    dataloader : Dataloader
        The dataloader that contains the data for the model to be evaluated on

    Returns
    -------
    accuracy : float
        The accuracy of the model
    errors : list[Any]
        The inputs for which the model had an incorrect prediction
    """
    with torch.no_grad():
        errors = []
        correct, total = 0, 0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = torch.argmax(model(inputs), dim = 1)
            correct += torch.sum(outputs == labels).item()

            total += 1

            for i in range(len(outputs)):
                if labels[i] != outputs[i]:
                    errors.append(inputs[i])

        return correct / len(inputs) / total, errors

def get_class_accs(model: nn.Module, dataloader: DataLoader):
    """
    Get the per-class accuracies of the given model on the given dataloader

    Parameters
    ----------
    model : nn.Module
        The model to evaluate
    dataloader : Dataloader
        The dataloader that contains the data for the model to be evaluated on

    Returns
    -------
    class_accs : defaultdict[Any, float]
        A mapping of classnames to the model's accuracy for that class
    """
    with torch.no_grad():
        # prepare to count predictions for each class
        correct_pred = defaultdict(lambda: 0)
        total_pred = defaultdict(lambda: 0)
        
        for data in dataloader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[label.item()] += 1
                total_pred[label.item()] += 1

        class_accs = defaultdict(lambda: 0)
        for classname, correct_count in correct_pred.items():
            class_accs[classname] = float(correct_count) / total_pred[classname]

        return class_accs
    

def num_params(model: nn.Module):
    """
    Get the number of trainable parameters in the given model

    Parameters
    ----------
    model : nn.Module
        The model to get the number of parameters of

    Returns
    -------
    n : int
        The number of trainable parameters in the given model
    """
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n
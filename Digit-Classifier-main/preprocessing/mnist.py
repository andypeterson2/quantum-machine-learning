from typing import Iterable, Optional

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import numpy as np

manualSeed = 999
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# Get the train, valid, and test dataloaders for mnist
def train_valid_test(classes: Iterable[int], batch_size : int = 1, n_train : int = 100, n_valid : int = 400):
    """
    Get the train, valid, and test dataloaders for mnist.
 
    Parameters
    ----------
    classes : Iterable[int]
        The classes that you want to include
    batch_size : int
        The number of samples in each batch
    n_train : int
        The number of training samples for each class
    n_valid : int
        The number of validation samples for each class
 
    Returns
    -------
    trainloader, validloader, testloader : tuple[DataLoader, DataLoader, DataLoader]
        The train, valid, and test dataloaders for mnist
    """

    # Use pre-defined torchvision function to load MNIST train data
    X_train = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    # Filter out labels (originally 0-9), leaving only labels 0 and 1
    idx = np.concatenate(
        [np.where(X_train.targets == c)[0][:n_train] for c in classes]
    )
    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]



    # Use pre-defined torchvision function to load MNIST train data
    X_test = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    idx = np.concatenate(
        [np.where(X_test.targets == c)[0][n_train:] for c in classes]
    )
    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]



    X_valid = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    idx = np.concatenate(
        [np.where(X_valid.targets == c)[0][:n_valid] for c in classes]
    )
    X_valid.data = X_valid.data[idx]
    X_valid.targets = X_valid.targets[idx]

    # Define torch dataloader with filtered data
    trainloader = DataLoader(X_train, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(X_valid, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(X_test, batch_size=batch_size, shuffle=True)

    return trainloader, validloader, testloader

# functions to show an image
def imshow(img: torch.Tensor):
    """
    Display the given torch tensor as an image.
 
    Parameters
    ----------
    img : torch.Tensor
        The tensor containing the image information
    """

    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def display_sample(dataloader: DataLoader, model: Optional[nn.Module] = None):
    """
    Display a random batch from the given dataloader.
 
    Parameters
    ----------
    dataloader : Dataloader
        The container from which to get the batch to display
    model : nn.Module | None
        The model whose prediction on the batch to display
    """

    # get some random training images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print("Labels:", ' '.join(f'{labels[j]:5d}' for j in range(len(labels))))

    # print predictions (if applicable)
    if not model: return
    preds = torch.argmax(model(images), dim=-1).tolist()
    print("Predictions:", ' '.join(f'{preds[j]:5d}' for j in range(len(preds))))
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

class Switchable(nn.Module):
    """
    The nn.Module that acts as a comparison wrapper for multiple specified models. 
    Given k different models, the Switchable object allows the user to switch between these k models easily.

    Attributes
    ----------
    models : list[nn.Module]
        The list of models to compare
    current : nn.Module
        The model that is currently in use

    Methods
    -------
    switch(index: int):
        Switch to using the model with the specified index
    from_model(model: nn.Module) -> Switchable:
        Construct a Switchable from a single model with multiple Switchable layers, each consisting of k models. 
        The new Switchable will be created from k new models, where model i will consist of the ith model from each of the
        Switchable layers and all non-Switchable layers.
    """

    # from list of models
    def __init__(self, models: list[nn.Module]):
        """
        Default constructor for Switchable
    
        Parameters
        ----------
        models : list[nn.Module]
            The list of models to compare
        """
        super().__init__()
        self.models = models
        self.current = models[0]

    # from single model with switchable layers
    def from_model(model: nn.Module):
        """
        Construct a Switchable from a single model with multiple Switchable layers, each consisting of k models. 
        The new Switchable will be created from k new models, where model i will consist of the ith model from each of the
        Switchable layers and all non-Switchable layers.

        Use this when you want to compare models that share many of the same layers but have a few different layers.
    
        Parameters
        ----------
        model : nn.Module
            The model from which to construct the Switchable
    
        Returns
        -------
        switchable_model : Switchable
            The final Switchable.
        """
        
        comparisons = 1
        for child in model.children():
            if isinstance(child, Switchable):
                comparisons = len(child.models)
                break

        models = [copy.deepcopy(model) for _ in range(comparisons)]
        for i, model in enumerate(models):
            for child in model.children():
                if isinstance(child, Switchable):
                    child.switch(i)
        return Switchable(models)

    def switch(self, index: int):
        """
        Switch to using the model with the specified index
    
        Parameters
        ----------
        index : int
            The index of the model to switch to using
        """

        self.current = self.models[index]

    def forward(self, x):
        x = self.current(x)
        return x

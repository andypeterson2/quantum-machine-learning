"""Reusable neural network layer building blocks.

Provides specialised layer types that can be used across any dataset plugin:

* :class:`Quadratic` — quadratic feature expansion layer.
* :class:`Polynomial` — polynomial basis layer (log-linear-exp).

Ported from the Digit-Classifier research codebase and generalised for
the plugin architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Quadratic(nn.Module):
    """Quadratic expansion layer.

    Takes input vector *x* and produces *y = W · z* where
    *z = concat(x^T · x, x)* — all pairwise quadratic products plus the
    original linear terms.

    The expanded dimension is ``input_dim * (input_dim + 1)``.

    Args:
        input_dim:  Dimension of the input vector.
        output_dim: Dimension of the output vector.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim * (input_dim + 1), output_dim)

    @staticmethod
    def expand(x: torch.Tensor) -> torch.Tensor:
        """Expand *x* into the quadratic feature vector *z*.

        Args:
            x: Tensor of shape ``(N, input_dim)``.

        Returns:
            Tensor of shape ``(N, input_dim * (input_dim + 1))``.
        """
        x = x.view(x.shape[0], 1, *x.shape[1:])
        xt = torch.transpose(x, 1, 2)
        # Append a column of ones so the product retains linear terms.
        x = torch.cat((x, torch.ones((x.shape[0], 1, 1), device=x.device)), dim=-1)
        z = torch.flatten(xt @ x, start_dim=1, end_dim=-1)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(Quadratic.expand(x))


class Polynomial(nn.Module):
    """Polynomial basis layer.

    Computes ``y = exp(W · log(|x| + 1))``, which creates polynomial-like
    feature transformations without explicit polynomial expansion.

    Args:
        input_dim:  Dimension of the input vector.
        output_dim: Dimension of the output vector.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.log(torch.abs(x) + 1)
        x = self.fc(x)
        x = torch.exp(x)
        return x

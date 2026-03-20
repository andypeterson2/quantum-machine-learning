"""Shared loss functions used across multiple dataset plugins.

Extracting loss functions here avoids cross-dataset imports.  Both the
MNIST and Iris SVM models can use :func:`multi_class_hinge_loss` without
either plugin depending on the other.
"""

from __future__ import annotations

import torch


def multi_class_hinge_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Compute the Crammer-Singer multi-class hinge loss.

    For each sample, every incorrect class whose score lies within *margin*
    of the correct class score incurs a penalty.  The ground-truth class is
    excluded from the sum via a multiplicative mask (no in-place ops, safe
    for autograd).

    Args:
        output: Raw model scores of shape ``(N, C)``.
        target: Ground-truth class indices of shape ``(N,)``.
        margin: Hinge margin (default ``1.0``).

    Returns:
        Mean hinge loss over the batch (scalar tensor).
    """
    n = output.size(0)
    correct_scores = output[torch.arange(n), target].unsqueeze(1)  # (N, 1)
    margins = (output - correct_scores + margin).clamp(min=0)       # (N, C)
    # Zero out the ground-truth class via multiplicative mask (autograd-safe)
    mask = torch.ones_like(margins)
    mask[torch.arange(n), target] = 0.0
    return (margins * mask).sum() / n

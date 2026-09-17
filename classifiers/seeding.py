"""One place to seed every RNG a training run touches.

Three libraries draw during a run: torch (weight initialisation, shuffling),
numpy, and the standard library's :mod:`random`. Seeding all three is what makes
a run repeatable, and doing it in one function keeps the exporter and the
:class:`~classifiers.trainer.Trainer` from drifting apart on which ones they
remember to seed.

Qiskit's Aer sampler is seeded indirectly: it draws its own simulator seed from
torch's generator when it is built (see
:class:`~classifiers.qiskit_layers._QCSampler`), so seeding here makes the
quantum models repeatable too.
"""

from __future__ import annotations

import random

import numpy as np
import torch

#: Seeds must fit in 32 bits — numpy rejects anything wider.
MAX_SEED = 2**32 - 1


def seed_everything(seed: int) -> None:
    """Seed :mod:`random`, numpy, and torch with *seed*.

    Args:
        seed: Any integer in ``[0, MAX_SEED]``.

    Raises:
        ValueError: If *seed* is outside that range.
    """
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}] (got {seed})")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

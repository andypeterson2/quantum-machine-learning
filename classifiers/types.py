"""Shared type aliases and protocols for the classifiers package.

Centralising these here prevents ``trainer`` and ``evaluator`` from each
defining their own identical ``StatusCallback`` alias, satisfying DRY and
making the shared callback contract explicit and discoverable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: Progress callback: a ``str`` ("Epoch 1/3 done") or a ``dict`` event with a ``"type"``
#: key. Trainer and Evaluator take one, so progress reporting is transport-agnostic.
StatusCallback = Callable[[str | dict[str, Any]], None]

"""Shared type aliases and protocols for the classifiers package.

Centralising these here prevents ``trainer`` and ``evaluator`` from each
defining their own identical ``StatusCallback`` alias, satisfying DRY and
making the shared callback contract explicit and discoverable.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, Union


# ── Structured training event ────────────────────────────────────────────────


class TrainingEvent(Protocol):
    """Protocol describing what structured SSE events look like.

    Any dict with at least a ``"type"`` key satisfies this protocol.
    The protocol exists purely for documentation and IDE support — callers
    can still pass plain dicts or strings.
    """

    def __getitem__(self, key: str) -> Any: ...
    def get(self, key: str, default: Any = None) -> Any: ...


#: Type alias for status/progress messages.
#:
#: A status message is either:
#:   - A ``str`` — human-readable progress text (e.g. ``"Epoch 1/3 done"``).
#:   - A ``dict[str, Any]`` — structured event with at least a ``"type"`` key
#:     (e.g. ``{"type": "history", "train_loss": 0.4, "val_accuracy": 0.9}``).
#:
#: Both :class:`~classifiers.trainer.Trainer` and
#: :class:`~classifiers.evaluator.Evaluator` accept an optional argument of
#: this type, decoupling progress reporting from any specific transport
#: (SSE queue, stdout, log file, etc.).
StatusCallback = Callable[[Union[str, dict[str, Any]]], None]

"""In-memory registry of trained models, namespaced by dataset.

The registry is the application's single source of truth for all loaded
models.  It is a plain Python object (not a Flask global or thread-local)
so it can be instantiated, tested, and replaced independently of the web
layer — consistent with the Dependency Inversion Principle.

Models are namespaced by dataset slug (e.g. ``"mnist"``, ``"iris"``), so
identically-named models from different datasets never collide.

All mutations to registry state go through explicit methods
(:meth:`~ModelRegistry.add`, :meth:`~ModelRegistry.remove`,
:meth:`~ModelRegistry.update_eval_result`) rather than direct attribute
assignment on :class:`ModelEntry`.  This gives a single auditable API
surface for state changes (Single Responsibility Principle).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .evaluator import EvalResult
from .base_model import BaseModel


@dataclass
class ModelEntry:
    """All data associated with a single registered model.

    Attributes:
        model:            Trained :class:`~classifiers.base_model.BaseModel` instance.
        model_type:       Registered display name of the architecture (e.g. ``"CNN"``).
        dataset:          Dataset slug this model belongs to (e.g. ``"mnist"``).
        epochs:           Number of epochs the model was trained for.
        batch_size:       Mini-batch size used during training.
        lr:               Learning rate used by the Adam optimiser.
        eval_result:      Test-set metrics, or ``None`` if not yet evaluated.
        training_history: Training curve data points (loss/accuracy per checkpoint).
        num_params:       Trainable parameter count (``None`` if unknown).
    """

    model: BaseModel
    model_type: str
    dataset: str
    epochs: int
    batch_size: int
    lr: float
    eval_result: EvalResult | None = None
    training_history: list[dict] = field(default_factory=list)
    num_params: int | None = None


class ModelRegistry:
    """Thread-safe* in-memory store mapping ``(dataset, name)`` pairs to :class:`ModelEntry` objects.

    Models are partitioned by dataset slug, so ``"Model 1"`` in MNIST is
    independent of ``"Model 1"`` in Iris.  An auto-incrementing counter
    per dataset generates default names ("Model 1", "Model 2", …).

    .. note::
        *Thread safety: individual method calls are atomic at the Python level
        due to the GIL, but compound check-then-act operations (e.g.
        ``if registry.get(d, n) is None: registry.add(d, n, ...)``) are not.
        For production multi-threaded use, wrap such compound operations in
        a lock.
    """

    def __init__(self) -> None:
        self._models: dict[str, dict[str, ModelEntry]] = {}
        self._counters: dict[str, int] = {}

    # ── Name generation ────────────────────────────────────────────────────────

    def next_name(self, dataset: str) -> str:
        """Return the next auto-generated model name for *dataset*.

        The counter never resets, so names stay unique for the process
        lifetime even if earlier models are removed.

        Args:
            dataset: Dataset slug (e.g. ``"mnist"``).

        Returns:
            A string of the form ``"Model N"``.
        """
        self._counters.setdefault(dataset, 0)
        self._counters[dataset] += 1
        return f"Model {self._counters[dataset]}"

    # ── Write operations ───────────────────────────────────────────────────────

    def add(
        self,
        dataset: str,
        name: str,
        model: BaseModel,
        model_type: str,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        """Add (or replace) a model entry in the registry.

        Args:
            dataset:    Dataset slug (e.g. ``"mnist"``).
            name:       Unique display name for this model within *dataset*.
            model:      Trained model instance.
            model_type: Registered architecture name (e.g. ``"CNN"``).
            epochs:     Number of training epochs.
            batch_size: Training batch size.
            lr:         Training learning rate.
        """
        self._models.setdefault(dataset, {})[name] = ModelEntry(
            model=model,
            model_type=model_type,
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )

    def remove(self, dataset: str, name: str) -> None:
        """Remove the model named *name* from *dataset*'s namespace.

        Silently does nothing if *name* or *dataset* is not present.

        Args:
            dataset: Dataset slug.
            name:    Display name of the model to remove.
        """
        if dataset in self._models:
            self._models[dataset].pop(name, None)

    def update_eval_result(self, dataset: str, name: str, result: EvalResult) -> None:
        """Attach evaluation metrics to an existing registry entry.

        Routing this update through the registry rather than assigning
        ``entry.eval_result`` directly from route code keeps all state
        changes behind a single, intentional API (SRP).

        Args:
            dataset: Dataset slug.
            name:    Display name of the model to update.
            result:  Evaluation metrics to attach.

        Raises:
            KeyError: If *name* is not present in *dataset*'s namespace.
        """
        ns = self._models.get(dataset, {})
        entry = ns.get(name)
        if entry is None:
            raise KeyError(f"Model '{name}' not found in dataset '{dataset}'")
        entry.eval_result = result

    # ── Read operations ────────────────────────────────────────────────────────

    def get(self, dataset: str, name: str) -> ModelEntry | None:
        """Return the :class:`ModelEntry` for *name* in *dataset*, or ``None``.

        Args:
            dataset: Dataset slug.
            name:    Display name of the model.

        Returns:
            The corresponding :class:`ModelEntry`, or ``None`` if absent.
        """
        return self._models.get(dataset, {}).get(name)

    def names(self, dataset: str) -> list[str]:
        """Return a snapshot of current model names for *dataset* in insertion order.

        Args:
            dataset: Dataset slug.

        Returns:
            A new list of name strings.
        """
        return list(self._models.get(dataset, {}).keys())

    def items(self, dataset: str) -> list[tuple[str, ModelEntry]]:
        """Return a snapshot of ``(name, entry)`` pairs for *dataset*.

        Args:
            dataset: Dataset slug.

        Returns:
            A new list of ``(str, ModelEntry)`` tuples.
        """
        return list(self._models.get(dataset, {}).items())

    def __len__(self) -> int:
        """Return the total number of models across all datasets."""
        return sum(len(ns) for ns in self._models.values())

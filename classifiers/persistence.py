"""Disk persistence for trained model checkpoints.

Separating file I/O into its own class keeps route handlers free of storage
concerns (SRP) and makes it trivial to swap the backend (OCP/DIP) — e.g.
substituting cloud storage without touching any route or registry code.

Checkpoints include a ``"dataset"`` field so that models are loaded back with
the correct architecture from the appropriate
:class:`~classifiers.dataset_plugin.DatasetPlugin`.  Existing ``.pt`` files
written before dataset support was added default to ``"mnist"`` for backward
compatibility.

The :class:`ModelPersistence` class is instantiated once inside
:func:`~classifiers.server.create_app` and stored in ``app.extensions`` so it
is accessible to routes via ``current_app`` without creating a module-level
global.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .plugin_registry import create_model

if TYPE_CHECKING:
    from .model_registry import ModelEntry


class ModelPersistence:
    """Saves and loads model checkpoints to/from a directory of ``.pt`` files.

    Each checkpoint is a plain Python dict serialised with :func:`torch.save`
    containing the model's ``state_dict``, its training hyper-parameters, and
    the dataset slug it was trained on.

    Args:
        models_dir: Absolute path to the folder where checkpoints are stored.
            The directory is created on first save if it does not exist.

    Example::

        store = ModelPersistence(Path("/app/models"))
        filename = store.save("My CNN", entry)
        files    = store.list_files()
        data     = store.load("my_cnn.pt")
    """

    def __init__(self, models_dir: Path) -> None:
        self._dir = models_dir

    # ── Public API ─────────────────────────────────────────────────────────────

    def save(self, name: str, entry: "ModelEntry") -> str:
        """Serialise *entry* to a ``.pt`` checkpoint file named after *name*.

        The checkpoint dict includes the ``"dataset"`` key so the model can
        be reloaded with the correct architecture.

        Args:
            name:  Human-readable model name used to derive the filename.
            entry: Registry entry containing the model and its hyper-parameters.

        Returns:
            The bare filename (not the full path) that was written.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        filename = self._safe_filename(name)
        checkpoint: dict[str, Any] = {
            "name": name,
            "dataset": entry.dataset,
            "model_type": entry.model_type,
            "state_dict": entry.model.state_dict(),
            "epochs": entry.epochs,
            "batch_size": entry.batch_size,
            "lr": entry.lr,
        }
        if entry.training_history:
            checkpoint["training_history"] = entry.training_history
        if entry.num_params is not None:
            checkpoint["num_params"] = entry.num_params
        torch.save(checkpoint, self._dir / filename)
        return filename

    def list_files(self) -> list[dict[str, Any]]:
        """Return metadata for every ``.pt`` file in the models directory.

        Files that cannot be parsed are still included with
        ``model_type="?"`` so the UI can show them without crashing.

        Returns:
            A list of dicts with keys: ``filename``, ``name``, ``dataset``,
            ``model_type``, ``epochs``, ``batch_size``, ``lr``.
        """
        if not self._dir.exists():
            return []

        results: list[dict[str, Any]] = []
        for p in sorted(self._dir.glob("*.pt")):
            try:
                data = torch.load(p, map_location="cpu", weights_only=False)
                results.append(
                    {
                        "filename": p.name,
                        "name": data.get("name", p.stem),
                        "dataset": data.get("dataset", "mnist"),
                        "model_type": data.get("model_type", "?"),
                        "epochs": data.get("epochs"),
                        "batch_size": data.get("batch_size"),
                        "lr": data.get("lr"),
                        "num_params": data.get("num_params"),
                    }
                )
            except Exception:
                results.append(
                    {"filename": p.name, "name": p.stem, "model_type": "?"}
                )
        return results

    def load(self, filename: str) -> dict[str, Any]:
        """Deserialise a checkpoint and return a ready-to-use model + metadata.

        The ``"dataset"`` field in the checkpoint determines which plugin's
        model factory is used to reconstruct the architecture.  Legacy files
        without this field default to ``"mnist"``.

        Args:
            filename: Bare filename (no directory component) ending in ``.pt``.

        Returns:
            A dict with keys: ``name`` (str), ``dataset`` (str),
            ``model_type`` (str), ``model`` (a
            :class:`~classifiers.base_model.BaseModel` instance with weights
            loaded and set to eval mode), ``epochs`` (int),
            ``batch_size`` (int), ``lr`` (float).

        Raises:
            ValueError:        If *filename* fails the safety check.
            FileNotFoundError: If the file does not exist in the models dir.
            KeyError:          If the checkpoint is missing ``model_type``.
        """
        self._validate_filename(filename)
        path = self._dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filename!r}")

        data = torch.load(path, map_location="cpu", weights_only=False)
        dataset: str = data.get("dataset", "mnist")
        model_type: str = data["model_type"]
        model = create_model(dataset, model_type)
        model.load_state_dict(data["state_dict"])
        model.eval()

        return {
            "name": data.get("name", path.stem),
            "dataset": dataset,
            "model_type": model_type,
            "model": model,
            "epochs": data.get("epochs", 0),
            "batch_size": data.get("batch_size", 64),
            "lr": data.get("lr", 1e-3),
            "training_history": data.get("training_history", []),
            "num_params": data.get("num_params"),
        }

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Convert an arbitrary model name into a safe ``.pt`` filename.

        Non-word characters (anything outside ``[a-zA-Z0-9_-]``) are replaced
        with underscores so the result is safe on all major operating systems.

        Args:
            name: Human-readable model name.

        Returns:
            A filename ending in ``.pt`` containing only word characters and
            hyphens.
        """
        return re.sub(r"[^\w\-]", "_", name) + ".pt"

    @staticmethod
    def _validate_filename(filename: str) -> None:
        """Reject filenames that could enable path-traversal attacks.

        Args:
            filename: The filename to validate.

        Raises:
            ValueError: If *filename* contains ``/``, ``..``, or does not end
                with ``.pt``.
        """
        if not filename.endswith(".pt") or "/" in filename or ".." in filename:
            raise ValueError(f"Invalid checkpoint filename: {filename!r}")

"""Dataset-agnostic model evaluation.

Evaluates a trained model on a supplied test-set :class:`~torch.utils.data.DataLoader`
and returns per-class and aggregate metrics.  The evaluator has **no** knowledge of
any specific dataset — it receives its data loader from the caller (typically a
route handler that obtains it from the active :class:`~classifiers.dataset_plugin.DatasetPlugin`).

This design follows the Dependency Inversion Principle: the evaluator depends on
the :class:`~torch.utils.data.DataLoader` abstraction, never on concrete dataset
libraries like ``torchvision``.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .base_model import BaseModel
from .types import StatusCallback

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Value object returned by :meth:`Evaluator.evaluate`.

    Attributes:
        accuracy:           Top-1 accuracy on the full test set (0.0 – 1.0).
        avg_loss:           Mean cross-entropy loss per sample.
        per_class_accuracy: Mapping from class label (str) to per-class
                            top-1 accuracy (0.0 – 1.0).
        num_params:         Trainable parameter count (``None`` if not computed).
    """

    accuracy: float
    avg_loss: float
    per_class_accuracy: dict[str, float] = field(default_factory=dict)
    num_params: int | None = None


class Evaluator:
    """Evaluates a trained model on any dataset's test set.

    The evaluator is fully stateless — the caller supplies the data loader,
    number of classes, and class labels.  Each call to :meth:`evaluate` returns
    a fresh :class:`EvalResult`.

    Example::

        plugin    = get_plugin("mnist")
        evaluator = Evaluator()
        result    = evaluator.evaluate(
            model, plugin.get_test_loader(1000),
            plugin.num_classes, plugin.class_labels,
            on_status=print,
        )
        print(f"Accuracy: {result.accuracy:.2%}")
    """

    def evaluate(
        self,
        model: BaseModel,
        test_loader: DataLoader,
        num_classes: int,
        class_labels: list[str],
        on_status: StatusCallback | None = None,
    ) -> EvalResult:
        """Run inference over *test_loader* and compute metrics.

        The model is set to eval mode automatically; the caller does not need
        to do this beforehand.

        Args:
            model:        Any trained :class:`~classifiers.base_model.BaseModel`.
            test_loader:  A :class:`~torch.utils.data.DataLoader` over the test set.
            num_classes:  Total number of target classes.
            class_labels: Ordered list of human-readable class names whose indices
                          correspond to the integer targets in *test_loader*.
            on_status:    Optional callback invoked with a progress string after
                          each batch.

        Returns:
            An :class:`EvalResult` with overall accuracy, average loss, and
            per-class accuracy keyed by class label.
        """

        def status(msg: str) -> None:
            if on_status:
                on_status(msg)

        status("Loading test set...")

        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        class_correct: dict[int, int] = {i: 0 for i in range(num_classes)}
        class_total: dict[int, int] = {i: 0 for i in range(num_classes)}

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                output = model(data)
                total_loss += F.cross_entropy(output, target, reduction="sum").item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)

                for t, p in zip(target, pred):
                    cls = t.item()
                    class_total[cls] += 1
                    if p.item() == cls:
                        class_correct[cls] += 1

                status(f"Evaluating... batch {batch_idx + 1}/{len(test_loader)}")

        per_class: dict[str, float] = {
            class_labels[i]: (
                class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
            )
            for i in range(num_classes)
        }

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        result = EvalResult(
            accuracy=correct / total if total > 0 else 0.0,
            avg_loss=total_loss / total if total > 0 else 0.0,
            per_class_accuracy=per_class,
            num_params=num_params,
        )
        status(f"Evaluation done — accuracy: {result.accuracy:.2%}")
        logger.info("Evaluation: accuracy=%.4f avg_loss=%.4f", result.accuracy, result.avg_loss)
        return result

    def ensemble_evaluate(
        self,
        models: list[BaseModel],
        test_loader: DataLoader,
        num_classes: int,
        class_labels: list[str],
        on_status: StatusCallback | None = None,
    ) -> EvalResult:
        """Majority-vote ensemble evaluation across multiple models.

        Each model votes for its argmax prediction. Ties are broken by the
        sum of logits across models.

        Args:
            models:       List of trained models.
            test_loader:  Test set data loader.
            num_classes:  Total number of target classes.
            class_labels: Ordered class label strings.
            on_status:    Optional progress callback.

        Returns:
            An :class:`EvalResult` with ensemble metrics.
        """

        def status(msg: str) -> None:
            if on_status:
                on_status(msg)

        status(f"Ensemble evaluation with {len(models)} models…")

        for m in models:
            m.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        class_correct: dict[int, int] = {i: 0 for i in range(num_classes)}
        class_total: dict[int, int] = {i: 0 for i in range(num_classes)}

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                # Collect votes and sum logits
                votes = torch.zeros(data.size(0), num_classes, dtype=torch.long)
                logit_sum = torch.zeros(data.size(0), num_classes)
                for m in models:
                    output = m(data)
                    logit_sum += output
                    preds = output.argmax(dim=1)
                    for i, p in enumerate(preds):
                        votes[i, p.item()] += 1

                # Break ties with summed logits
                max_votes = votes.max(dim=1, keepdim=True).values
                tied = votes == max_votes
                # Zero out non-tied logits, then argmax
                masked_logits = logit_sum.clone()
                masked_logits[~tied] = float("-inf")
                ensemble_pred = masked_logits.argmax(dim=1)

                total_loss += F.cross_entropy(logit_sum, target, reduction="sum").item()
                correct += ensemble_pred.eq(target).sum().item()
                total += len(target)

                for t, p in zip(target, ensemble_pred):
                    cls = t.item()
                    class_total[cls] += 1
                    if p.item() == cls:
                        class_correct[cls] += 1

                status(f"Ensemble eval… batch {batch_idx + 1}/{len(test_loader)}")

        per_class = {
            class_labels[i]: (
                class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
            )
            for i in range(num_classes)
        }

        result = EvalResult(
            accuracy=correct / total if total > 0 else 0.0,
            avg_loss=total_loss / total if total > 0 else 0.0,
            per_class_accuracy=per_class,
        )
        status(f"Ensemble done — accuracy: {result.accuracy:.2%}")
        return result

    def ablation_evaluate(
        self,
        model: BaseModel,
        test_loader: DataLoader,
        num_classes: int,
        class_labels: list[str],
        on_status: StatusCallback | None = None,
    ) -> dict[str, EvalResult]:
        """Ablation study: zero out each layer's weights and re-evaluate.

        For each named parameter group (layer), creates a deep copy of the
        model with that layer's parameters zeroed out, then evaluates on the
        test set. This measures each layer's contribution to accuracy.

        Args:
            model:        The trained model to ablate.
            test_loader:  Test set data loader.
            num_classes:  Total number of target classes.
            class_labels: Ordered class label strings.
            on_status:    Optional progress callback.

        Returns:
            A dict mapping layer name to :class:`EvalResult` with that
            layer zeroed out.
        """

        def status(msg: str | dict) -> None:
            if on_status:
                on_status(msg)

        # Discover unique layer prefixes from named parameters
        layer_names: list[str] = []
        seen: set[str] = set()
        for name, _ in model.named_parameters():
            prefix = name.rsplit(".", 1)[0]
            if prefix not in seen:
                seen.add(prefix)
                layer_names.append(prefix)

        status(f"Ablation study: {len(layer_names)} layers to test")

        # Get baseline accuracy
        baseline = self.evaluate(model, test_loader, num_classes, class_labels)

        results: dict[str, EvalResult] = {}
        for layer_name in layer_names:
            status(f"Ablating layer: {layer_name}")
            ablated = copy.deepcopy(model)
            # Zero out parameters belonging to this layer
            for name, param in ablated.named_parameters():
                if name.startswith(layer_name + ".") or name == layer_name:
                    param.data.zero_()

            result = self.evaluate(ablated, test_loader, num_classes, class_labels)
            results[layer_name] = result

            drop = baseline.accuracy - result.accuracy
            status({
                "type": "ablation_result",
                "layer": layer_name,
                "accuracy": round(result.accuracy, 4),
                "drop": round(drop, 4),
            })

        status("Ablation study complete!")
        return results

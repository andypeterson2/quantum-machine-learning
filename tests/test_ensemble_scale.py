"""An ensemble member's influence must not depend on its logit scale.

The models here are trained to different objectives: cross-entropy models emit
logits of one size, hinge-trained SVMs emit scores several times larger, and a
QVC is bounded to [-1, 1]. The ensemble used to sum those raw scores to break
ties and to compute its loss, so the loudest model settled every tie and the
loss moved with the ensemble's composition rather than its quality.

These tests use deliberately mismatched scales, which is what the real mixture
looks like.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from classifiers.base_model import BaseModel
from classifiers.evaluator import Evaluator

CLASS_LABELS = ["a", "b"]
NUM_CLASSES = 2


class _Constant(BaseModel):
    """Votes for a fixed class, with a configurable confidence scale."""

    name = "constant"
    description = "test double"

    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("_logits", torch.tensor([logits], dtype=torch.float32))
        self.unused = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._logits.expand(x.size(0), -1)


def _loader(n: int = 8):
    """One batch of n samples, every label class 0."""
    return [(torch.randn(n, 4), torch.zeros(n, dtype=torch.long))]


def _evaluate(models, loader=None):
    return Evaluator().ensemble_evaluate(
        models, loader or _loader(), NUM_CLASSES, CLASS_LABELS
    )


def test_one_loud_model_cannot_outweigh_a_confident_side() -> None:
    """A 2-2 tie where the raw sums and the distributions disagree.

    Two models back class 0 at 95% each; class 1 is backed by one model that is
    barely leaning (52%) and one whose logits are enormous. Summing raw scores
    lets that one model's scale decide for everybody (6 against 50). Averaging
    distributions gives class 0, which is what the members actually believe.
    """
    result = _evaluate(
        [
            _Constant([3.0, 0.0]),
            _Constant([3.0, 0.0]),
            _Constant([0.0, 50.0]),   # the loud one
            _Constant([0.0, 0.1]),
        ]
    )
    assert result.accuracy == 1.0


def test_tie_break_ignores_a_uniform_rescaling() -> None:
    """Doubling every score changes nothing a well-posed ensemble reports."""
    base = [_Constant([2.0, 0.0]), _Constant([0.0, 3.0])]
    scaled = [_Constant([4.0, 0.0]), _Constant([0.0, 6.0])]
    assert _evaluate(base).accuracy == _evaluate(scaled).accuracy


def test_loss_does_not_grow_with_the_number_of_models() -> None:
    """Summed logits made the loss scale with membership; a mean does not.

    Three identical models describe exactly the distribution one of them does,
    so the ensemble's loss must not move.
    """
    model = _Constant([3.0, 0.0])
    loader = _loader()
    one = _evaluate([model, _Constant([3.0, 0.0])], loader)
    three = _evaluate([model] + [_Constant([3.0, 0.0]) for _ in range(2)], loader)
    assert three.avg_loss == pytest.approx(one.avg_loss, abs=1e-6)


def test_loss_is_a_negative_log_likelihood() -> None:
    """Identical members: the loss is -log of the probability they agree on."""
    logits = [3.0, 0.0]
    expected = -torch.log_softmax(torch.tensor(logits), dim=0)[0].item()
    result = _evaluate([_Constant(logits), _Constant(logits)])
    assert result.avg_loss == pytest.approx(expected, abs=1e-5)


def test_majority_still_decides_before_the_tie_break() -> None:
    """The tie-break only applies to ties: two quiet votes beat one loud one."""
    result = _evaluate(
        [_Constant([1.0, 0.0]), _Constant([1.0, 0.0]), _Constant([0.0, 50.0])]
    )
    assert result.accuracy == 1.0

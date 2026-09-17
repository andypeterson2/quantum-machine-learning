"""Seeded training runs repeat; unseeded ones stay free-running.

Before this, only the web exporter seeded anything, so a run through the API or
the Trainer could not be reproduced — the reason the distillation numbers in the
history cannot be regenerated today.
"""

from __future__ import annotations

import pytest
import torch

from classifiers.datasets.iris.models import IrisLinear
from classifiers.seeding import MAX_SEED, seed_everything
from classifiers.server import create_app
from classifiers.trainer import Trainer


@pytest.fixture
def app(tmp_path):
    return create_app(models_dir=tmp_path)


@pytest.fixture
def client(app):
    return app.test_client()


class _Loader(list):
    """A list of batches with the one DataLoader attribute Trainer reads."""

    batch_size = 8


def _iris_loader(n_batches=3, batch_size=8):
    """Synthetic 4-feature / 3-class batches, fixed so only seeding varies."""
    gen = torch.Generator().manual_seed(1234)
    return _Loader(
        (
            torch.randn(batch_size, 4, generator=gen),
            torch.randint(0, 3, (batch_size,), generator=gen),
        )
        for _ in range(n_batches)
    )


def _train(seed):
    return Trainer(
        model_cls=IrisLinear, train_loader=_iris_loader(), dataset="iris", epochs=1, seed=seed
    ).train()


def _weights(result):
    return torch.cat([p.detach().flatten() for p in result.model.parameters()])


class TestSeedEverything:
    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="seed must be in"):
            seed_everything(MAX_SEED + 1)

    def test_seeds_torch(self):
        seed_everything(7)
        first = torch.randn(3)
        seed_everything(7)
        assert torch.equal(first, torch.randn(3))


class TestSeededTraining:
    def test_same_seed_gives_identical_weights(self):
        assert torch.equal(_weights(_train(11)), _weights(_train(11)))

    def test_different_seeds_diverge(self):
        assert not torch.equal(_weights(_train(11)), _weights(_train(12)))

    def test_seed_is_reported(self):
        assert _train(11).seed == 11

    def test_unseeded_runs_report_no_seed(self):
        result = Trainer(
            model_cls=IrisLinear, train_loader=_iris_loader(), dataset="iris", epochs=1
        ).train()
        assert result.seed is None

    def test_unseeded_runs_are_left_free_running(self):
        """Training must not quietly seed the global RNG for everything after it."""
        seed_everything(3)
        expected = torch.randn(2)
        seed_everything(3)
        Trainer(
            model_cls=IrisLinear, train_loader=_iris_loader(), dataset="iris", epochs=1
        ).train()
        assert not torch.equal(expected, torch.randn(2))


class TestSeedOverHttp:
    def test_seed_is_echoed(self, client):
        resp = client.post(
            "/d/iris/train/sync", json={"model_type": "Linear", "epochs": 1, "seed": 99}
        )
        assert resp.status_code == 200
        assert resp.get_json()["seed"] == 99

    def test_absent_seed_is_absent_from_the_result(self, client):
        resp = client.post("/d/iris/train/sync", json={"model_type": "Linear", "epochs": 1})
        assert resp.status_code == 200
        assert "seed" not in resp.get_json()

    @pytest.mark.parametrize("seed", [-1, MAX_SEED + 1, "abc"])
    def test_out_of_range_seed_is_400(self, client, seed):
        resp = client.post(
            "/d/iris/train/sync", json={"model_type": "Linear", "epochs": 1, "seed": seed}
        )
        assert resp.status_code == 400


class TestQuantumSampling:
    """The Aer sampler takes its simulator seed from torch's generator, so a
    seeded run covers the quantum models too."""

    def test_sampler_seed_follows_torch(self):
        pytest.importorskip("qiskit_aer", reason="qiskit-aer not installed")
        from classifiers.qiskit_layers import _QCSampler

        seed_everything(5)
        first = _QCSampler().seed
        seed_everything(5)
        assert _QCSampler().seed == first

    def test_seeded_sampling_repeats(self):
        pytest.importorskip("qiskit_aer", reason="qiskit-aer not installed")
        from classifiers.qiskit_layers import _ExampleCircuit

        seed_everything(5)
        circuit = _ExampleCircuit(3)
        weights, inputs = [0.1, 0.2, 0.3, 0.4], [0.5, -0.5, 0.25]
        assert circuit.run(weights, inputs).tolist() == circuit.run(weights, inputs).tolist()

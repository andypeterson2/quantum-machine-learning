"""BB84 quantum-key-distribution session simulator.

Generates the dataset the ``bb84`` plugin trains on: whole BB84 sessions,
summarised by the two observables an operator actually sees — the quantum bit
error rate (QBER) and the sifted-key rate — labelled *clean* or *eavesdropped*
(intercept-resend attack).

The channel physics is a NumPy port of the quantum-video-chat project's
``SimulatedQuantumChannel`` (Poisson photon source, fiber attenuation at
0.2 dB/km, avalanche-photodiode detector efficiency, intercept-resend Eve),
with two extensions that make the classification task honest rather than
trivial:

* **channel noise** — clean sessions carry a per-pulse misalignment flip
  probability, so their QBER is not identically zero;
* **partial, lossy interception** — Eve intercepts only a fraction of the
  arriving photons and her own detector is imperfect, so eavesdropped
  sessions sweep a QBER range that overlaps the noisy-clean regime near the
  protocol's 11 % abort threshold, and interception also depresses the
  sifted-key rate (photons Eve fails to register never reach Bob).

Everything is driven by an explicit ``numpy.random.Generator``, so a fixed
seed reproduces the exact dataset — which is what lets the web-export drift
check re-derive the committed artifacts in CI with no cached data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Standard telecom-fiber attenuation, as in the video-chat channel model.
ATTENUATION_DB_PER_KM = 0.2

#: Raw pulses per simulated session (matches the video chat's protocol scale).
NUM_PULSES = 4096

# ── Per-session parameter regimes ─────────────────────────────────────────────
# Channel parameters vary session-to-session so the sifted-key rate carries
# real information about the link, not a constant.
FIBER_KM_RANGE = (0.5, 2.0)
SOURCE_INTENSITY_RANGE = (0.4, 0.6)
DETECTOR_EFFICIENCY_RANGE = (0.4, 0.6)

#: Clean sessions: misalignment/dark-count flip probability (QBER ~ 0–6 %).
CLEAN_MISALIGNMENT_RANGE = (0.0, 0.06)
#: Eavesdropped sessions ride a quieter channel but carry Eve's errors.
EVE_MISALIGNMENT_RANGE = (0.0, 0.03)
#: Fraction of arriving photons Eve intercepts (full attack = 1.0).
EVE_INTERCEPT_RANGE = (0.25, 1.0)
#: Eve's own detector efficiency — intercepted photons she misses are lost.
EVE_DETECTOR_RANGE = (0.75, 0.95)


@dataclass(frozen=True)
class SessionConfig:
    """One session's channel + adversary parameters.

    Attributes:
        intercept_fraction: Fraction of arriving photons Eve intercepts.
        misalignment: Per-detected-pulse bit-flip probability (channel noise).
        fiber_length_km: Fiber length (attenuation at 0.2 dB/km).
        source_intensity: Mean photon number per weak-coherent pulse (μ).
        detector_efficiency: Bob's APD detection efficiency.
        eve_detector_efficiency: Eve's detector efficiency; photons she
            intercepts but fails to register never reach Bob.
        num_pulses: Raw pulses in the session.
    """

    intercept_fraction: float = 0.0
    misalignment: float = 0.0
    fiber_length_km: float = 1.0
    source_intensity: float = 0.5
    detector_efficiency: float = 0.5
    eve_detector_efficiency: float = 1.0
    num_pulses: int = NUM_PULSES


def simulate_session(
    rng: np.random.Generator, config: SessionConfig | None = None
) -> tuple[float, float]:
    """Simulate one BB84 session; return ``(qber, sifted_key_rate)``.

    Args:
        rng: Random generator driving every stochastic step.
        config: Channel + adversary parameters (default: clean 1 km link).

    Returns:
        ``(qber, sifted_key_rate)`` — error rate over the sifted bits, and
        sifted bits per raw pulse.
    """
    config = config or SessionConfig()
    n = config.num_pulses
    # Poisson source: probability of at least one photon in the pulse.
    photon = rng.random(n) < 1.0 - np.exp(-config.source_intensity)
    # Fiber attenuation.
    transmittance = 10.0 ** (-(ATTENUATION_DB_PER_KM * config.fiber_length_km) / 10.0)
    arrived = photon & (rng.random(n) < transmittance)

    # Intercept-resend Eve on a fraction of the arriving photons.
    intercepted = arrived & (rng.random(n) < config.intercept_fraction)
    eve_registered = intercepted & (rng.random(n) < config.eve_detector_efficiency)
    lost_to_eve = intercepted & ~eve_registered
    # Eve measures in a random basis; a wrong-basis interception leaves Bob
    # (measuring in Alice's basis after sifting) with a coin-flip — i.e. a
    # bit error half the time. Same model as the video chat's channel.
    eve_wrong_basis = eve_registered & (rng.random(n) < 0.5)
    eve_flip = eve_wrong_basis & (rng.random(n) < 0.5)

    # Bob's detector.
    detected = arrived & ~lost_to_eve & (rng.random(n) < config.detector_efficiency)
    # Channel noise on detected pulses.
    noise_flip = rng.random(n) < config.misalignment

    # Basis reconciliation: independent basis choices agree half the time.
    bases_match = rng.random(n) < 0.5
    sifted = detected & bases_match
    n_sifted = int(sifted.sum())
    if n_sifted == 0:
        return 0.0, 0.0

    errors = (eve_flip ^ noise_flip) & sifted
    qber = float(errors.sum()) / n_sifted
    return qber, n_sifted / n


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    """Draw one value uniformly from an inclusive-exclusive range tuple."""
    return float(rng.uniform(bounds[0], bounds[1]))


def simulate_labeled_session(
    rng: np.random.Generator, *, eavesdropped: bool
) -> tuple[float, float]:
    """Simulate one session with regime-appropriate random parameters."""
    kwargs = {
        "fiber_length_km": _uniform(rng, FIBER_KM_RANGE),
        "source_intensity": _uniform(rng, SOURCE_INTENSITY_RANGE),
        "detector_efficiency": _uniform(rng, DETECTOR_EFFICIENCY_RANGE),
    }
    if eavesdropped:
        kwargs["intercept_fraction"] = _uniform(rng, EVE_INTERCEPT_RANGE)
        kwargs["eve_detector_efficiency"] = _uniform(rng, EVE_DETECTOR_RANGE)
        kwargs["misalignment"] = _uniform(rng, EVE_MISALIGNMENT_RANGE)
    else:
        kwargs["misalignment"] = _uniform(rng, CLEAN_MISALIGNMENT_RANGE)
    return simulate_session(rng, SessionConfig(**kwargs))


def generate_dataset(n_sessions: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate a balanced, seeded dataset of simulated BB84 sessions.

    Args:
        n_sessions: Total sessions (alternating clean / eavesdropped).
        seed: RNG seed; the same seed reproduces the same dataset exactly.

    Returns:
        ``(X, y)`` — features ``(n_sessions, 2)`` float32 in the order
        ``(qber, sifted_key_rate)``, and int64 labels (0 = clean,
        1 = eavesdropped).
    """
    rng = np.random.default_rng(seed)
    features = np.empty((n_sessions, 2), dtype=np.float32)
    labels = np.empty(n_sessions, dtype=np.int64)
    for i in range(n_sessions):
        eavesdropped = i % 2 == 1
        qber, rate = simulate_labeled_session(rng, eavesdropped=eavesdropped)
        features[i, 0] = qber
        features[i, 1] = rate
        labels[i] = int(eavesdropped)
    return features, labels

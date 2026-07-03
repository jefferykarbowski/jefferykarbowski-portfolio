"""Generative pattern algorithms: Euclidean rhythms, Markov walks in timbre space,
golden-ratio song structure, Fibonacci subdivision."""

from __future__ import annotations

import numpy as np

PHI = (1 + 5**0.5) / 2
FIB = [1, 2, 3, 5, 8, 13, 21]


def euclidean_rhythm(pulses: int, steps: int, rotation: int = 0) -> list[bool]:
    """Distribute `pulses` hits as evenly as possible across `steps` (Bjorklund)."""
    if pulses <= 0:
        return [False] * steps
    pulses = min(pulses, steps)
    pattern = [((i * pulses) % steps) < pulses for i in range(steps)]
    return pattern[-rotation % steps :] + pattern[: -rotation % steps]


def markov_walk(
    features: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
    temperature: float = 1.0,
    start: int | None = None,
) -> list[int]:
    """Random walk over items where transition probability decays with
    timbre-space distance — nearby-sounding slices chain together, with
    occasional temperature-driven leaps."""
    n = len(features)
    if n == 0:
        return []
    if n == 1:
        return [0] * n_steps
    diff = features[:, None, :] - features[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=-1))
    scale = np.median(dist[dist > 0]) * max(temperature, 1e-3)
    P = np.exp(-dist / scale)
    np.fill_diagonal(P, 0.05)  # discourage (not forbid) repeats
    P /= P.sum(axis=1, keepdims=True)
    state = int(start) if start is not None else int(rng.integers(n))
    out = []
    for _ in range(n_steps):
        out.append(state)
        state = int(rng.choice(n, p=P[state]))
    return out


def golden_structure(total_bars: int) -> list[tuple[str, int, int]]:
    """Split the track into sections at golden-ratio points (powers of 1/phi),
    quantized to 2-bar boundaries. Returns (name, start_bar, end_bar)."""
    marks = [PHI**-4, PHI**-2, PHI**-1, PHI**-1 + PHI**-4, 1 - PHI**-6]
    names = ["intro", "build", "drop", "breakdown", "drop2", "outro"]
    q = lambda x: int(round(x * total_bars / 2)) * 2
    cuts = [0] + [min(q(m), total_bars) for m in marks] + [total_bars]
    sections = []
    for name, a, b in zip(names, cuts, cuts[1:]):
        if b > a:
            sections.append((name, a, b))
    return sections


def fibonacci_density(n_bars: int, lo: int = 2, hi: int = 13) -> list[int]:
    """Hit counts that climb the Fibonacci sequence across a section — used to
    make buildups accelerate non-linearly."""
    ladder = [f for f in FIB if lo <= f <= hi] or [lo]
    idx = np.linspace(0, len(ladder) - 1, n_bars).round().astype(int)
    return [ladder[i] for i in idx]

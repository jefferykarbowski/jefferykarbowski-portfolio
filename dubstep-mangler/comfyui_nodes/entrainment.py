"""Brainwave-entrainment and experimental 'vibration' signal generators.

Everything here is pure numpy so it can be unit-tested without ComfyUI/torch.

Honesty note: binaural beats and isochronic tones are established *audio
techniques* (the acoustics are real and implemented faithfully below), but
scientific evidence that they alter mental states is mixed. The experimental
generators (phi spirals, Fibonacci gates, Schumann locks) are novel sound
design built on mathematical structure — presented as art, not medicine.
Binaural beats require headphones; the effect is created inside the listener's
auditory system from the L/R frequency difference.
"""

from __future__ import annotations

import numpy as np

PHI = (1 + 5**0.5) / 2
FIB = [1, 1, 2, 3, 5, 8, 13, 21, 34]
SCHUMANN_HZ = 7.83  # fundamental resonance of the Earth-ionosphere cavity

# Classic brainwave bands (beat/pulse rate targets, Hz)
BRAINWAVE_PRESETS = {
    "delta (deep sleep, 2 Hz)": 2.0,
    "theta (meditation, 6 Hz)": 6.0,
    "alpha (relaxation, 10 Hz)": 10.0,
    "beta (focus, 18 Hz)": 18.0,
    "gamma (insight, 40 Hz)": 40.0,
    "schumann (earth, 7.83 Hz)": SCHUMANN_HZ,
}

# Solfeggio carrier frequencies (traditional set)
SOLFEGGIO_PRESETS = {
    "UT 396 Hz (liberation)": 396.0,
    "RE 417 Hz (change)": 417.0,
    "MI 528 Hz (transformation)": 528.0,
    "FA 639 Hz (connection)": 639.0,
    "SOL 741 Hz (expression)": 741.0,
    "LA 852 Hz (intuition)": 852.0,
}


def _t(n: int, sr: int) -> np.ndarray:
    return np.arange(n) / sr


def _phase(freq: np.ndarray | float, n: int, sr: int) -> np.ndarray:
    """Integrated phase for (possibly time-varying) frequency."""
    f = np.broadcast_to(np.asarray(freq, dtype=np.float64), (n,))
    return 2 * np.pi * np.cumsum(f) / sr


def _edge_fade(x: np.ndarray, sr: int, ms: float = 30.0) -> np.ndarray:
    f = min(int(ms / 1000 * sr), x.shape[-1] // 2)
    if f > 0:
        x[..., :f] *= np.linspace(0, 1, f)
        x[..., -f:] *= np.linspace(1, 0, f)
    return x


def binaural_beats(
    n: int,
    sr: int,
    carrier_hz: float = 220.0,
    beat_start_hz: float = 10.0,
    beat_end_hz: float | None = None,
) -> np.ndarray:
    """Stereo (2, n): left ear gets the carrier, right ear gets carrier + beat.
    The beat frequency can sweep (e.g. alpha 10 Hz down to theta 6 Hz for a
    guided descent). Headphones required."""
    beat_end_hz = beat_start_hz if beat_end_hz is None else beat_end_hz
    beat = np.linspace(beat_start_hz, beat_end_hz, n)
    left = np.sin(_phase(carrier_hz, n, sr))
    right = np.sin(_phase(carrier_hz + beat, n, sr))
    return _edge_fade(np.stack([left, right]).astype(np.float32), sr)


def isochronic_tones(
    n: int,
    sr: int,
    carrier_hz: float = 528.0,
    rate_start_hz: float = 10.0,
    rate_end_hz: float | None = None,
    duty: float = 0.5,
    softness: float = 0.15,
) -> np.ndarray:
    """Mono (n,): a carrier switched fully on/off at the pulse rate — the
    speaker-friendly sibling of binaural beats. `softness` rounds the pulse
    edges (raised-cosine) to avoid clicks; `duty` is the on-fraction."""
    rate_end_hz = rate_start_hz if rate_end_hz is None else rate_end_hz
    rate = np.linspace(rate_start_hz, rate_end_hz, n)
    cycle = (np.cumsum(rate) / sr) % 1.0  # 0..1 position within each pulse
    gate = np.clip((duty - np.abs(cycle - duty / 2) * 2) / max(softness * duty, 1e-4), 0, 1)
    gate = 0.5 - 0.5 * np.cos(np.pi * gate)  # raised-cosine edges
    tone = np.sin(_phase(carrier_hz, n, sr))
    return _edge_fade((tone * gate).astype(np.float32), sr)


def pulse_gate(x: np.ndarray, sr: int, rate_hz: float, duty: float = 0.6) -> np.ndarray:
    """Apply an isochronic on/off gate to existing audio (any shape (..., n))."""
    n = x.shape[-1]
    return (x * isochronic_gate_env(n, sr, rate_hz, duty)).astype(np.float32)


def isochronic_gate_env(n: int, sr: int, rate_hz: float, duty: float = 0.6) -> np.ndarray:
    cycle = (np.arange(n) * rate_hz / sr) % 1.0
    gate = np.clip((duty - np.abs(cycle - duty / 2) * 2) / max(0.1 * duty, 1e-4), 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * gate)


# --- experimental generators ------------------------------------------------


def phi_spiral_stack(
    n: int,
    sr: int,
    base_hz: float = 111.0,
    n_partials: int = 8,
    shimmer: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    """Inharmonic drone: partials at base * phi^k (golden-ratio spacing) with
    amplitudes 1/phi^k, each slowly amplitude-modulated at a phi-derived
    sub-hertz rate. Because the partials never align harmonically, the sound
    slowly 'rotates' without ever repeating — a spectrum you won't find in
    conventional (harmonic-series) instruments."""
    rng = np.random.default_rng(seed)
    t = _t(n, sr)
    out = np.zeros(n)
    for k in range(n_partials):
        f = base_hz * PHI**k
        if f > sr / 2 * 0.9:
            break
        amp = PHI**-k
        am_rate = (PHI**-1) * (k + 1) / 10.0 * shimmer  # 0.06..~0.5 Hz
        am = 0.6 + 0.4 * np.sin(2 * np.pi * am_rate * t + rng.uniform(0, 2 * np.pi))
        out += amp * am * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    out /= np.abs(out).max() + 1e-9
    return _edge_fade(out.astype(np.float32), sr, ms=200)


def fibonacci_gate(x: np.ndarray, sr: int, step_s: float = 0.25) -> np.ndarray:
    """Gate audio open only on Fibonacci-numbered steps of each 34-step cycle
    (1,1,2,3,5,8,13,21 -> positions mod 34). A rhythm with accelerating gaps
    that resolves each cycle — pattern first, groove second."""
    n = x.shape[-1]
    step_len = max(int(step_s * sr), 1)
    positions = {f % 34 for f in FIB}
    steps = (np.arange(n) // step_len) % 34
    env = np.isin(steps, list(positions)).astype(np.float64)
    # 10ms raised-cosine smoothing so the gate never clicks
    k = max(int(0.01 * sr), 1)
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(2 * k) / (2 * k))
    env = np.convolve(env, win / win.sum(), mode="same")
    return (x * env).astype(np.float32)


def schumann_lock(x: np.ndarray, sr: int, depth: float = 0.35) -> np.ndarray:
    """Amplitude-modulate audio at the Schumann resonance (7.83 Hz) plus a
    quieter golden-ratio partner (7.83 * phi ≈ 12.67 Hz). The two rates beat
    against each other at ~4.8 Hz — a modulation-of-modulation pattern."""
    n = x.shape[-1]
    t = _t(n, sr)
    am = 1.0 - depth * 0.5 * (
        (1 + np.sin(2 * np.pi * SCHUMANN_HZ * t))
        + 0.5 * (1 + np.sin(2 * np.pi * SCHUMANN_HZ * PHI * t))
    ) / 1.5
    return (x * am).astype(np.float32)


def breath_pacer(x: np.ndarray, sr: int, breaths_per_min: float = 5.5, depth: float = 0.5) -> np.ndarray:
    """Swell the audio at a slow-breathing pace (default 5.5/min, the rate
    used in coherent-breathing practice) with a longer exhale than inhale."""
    n = x.shape[-1]
    t = _t(n, sr)
    cycle = (t * breaths_per_min / 60.0) % 1.0
    inhale = np.clip(cycle / 0.4, 0, 1)  # 40% in
    exhale = np.clip((1 - cycle) / 0.6, 0, 1)  # 60% out
    env = 1.0 - depth + depth * np.minimum(inhale, exhale)
    return (x * env).astype(np.float32)

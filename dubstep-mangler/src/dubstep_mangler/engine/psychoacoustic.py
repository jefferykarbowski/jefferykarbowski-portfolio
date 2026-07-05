"""Psychoacoustic primitives for the Hypnagogia Engine.

The governing idea: don't ADD an audible beat — make the *music itself* pulse at
the target rate across several perceptual channels at once (loudness, stereo
position, timbre), each subtle enough to stay subliminal but reinforcing each
other. And structure the material so the cortex can't extract a period or meter
(incommensurate φ/√2 layers, 1/f fractal melody, Risset eternal glides), so it
never builds a predictive model it can then tune out. Un-modelable = un-skippable.

Pure numpy/scipy.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

PHI = (1 + 5**0.5) / 2
SQRT2 = 2**0.5
SQRT3 = 3**0.5


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3 - 2 * t)


def rate_curve(n: int, sr: int, start_hz: float, end_hz: float, drift_hz: float, seed: int) -> np.ndarray:
    """Instantaneous entrainment rate: an eased iso-principle descent from
    start_hz (alpha) to end_hz (theta) with a slow anti-habituation wander."""
    prog = _smoothstep(np.linspace(0, 1, n))
    rate = start_hz + (end_hz - start_hz) * prog
    if drift_hz > 0:
        rng = np.random.default_rng(seed)
        n_ctrl = max(int(n / sr * 0.05) + 2, 2)
        walk = np.interp(np.arange(n), np.linspace(0, n, n_ctrl), rng.uniform(-1, 1, n_ctrl))
        rate = rate + drift_hz * walk / (np.abs(walk).max() + 1e-9)
    return np.clip(rate, 0.5, 45.0)


def _phase(rate: np.ndarray, sr: int) -> np.ndarray:
    return 2 * np.pi * np.cumsum(rate) / sr


def embedded_entrainment(
    music: np.ndarray,
    sr: int,
    rate: np.ndarray,
    tremolo_depth: float = 0.10,
    pan_depth: float = 0.35,
    filter_depth: float = 0.25,
    seed: int = 0,
) -> np.ndarray:
    """Bake the entrainment rate into musical motion. Three subliminal carriers,
    all sharing `rate` (so they entrain together) but phase-spread (so none is
    individually obvious):

      * tremolo  — gentle loudness breathing (monaural beat percept)
      * autopan  — equal-power rotary in the stereo field (interaural percept)
      * filter   — slow lowpass sweep (spectral percept)

    Input/return stereo (2, n)."""
    st = music if music.ndim == 2 else np.stack([music, music])
    n = st.shape[-1]
    rate = np.resize(rate, n)
    ph = _phase(rate, sr)

    # tremolo (loudness)
    trem = 1.0 - tremolo_depth * 0.5 * (1 + np.sin(ph))
    out = st * trem

    # autopan (equal power), quarter-cycle offset from tremolo
    pan = 0.5 * (1 + np.sin(ph + np.pi / 2))
    gL = 1 + pan_depth * (SQRT2 * np.cos(pan * np.pi / 2) - 1)
    gR = 1 + pan_depth * (SQRT2 * np.sin(pan * np.pi / 2) - 1)
    out = np.stack([out[0] * gL, out[1] * gR])

    # spectral pulse (timbre) — time-varying lowpass, half-cycle offset
    if filter_depth > 0:
        cutoff = 1800 * (1 + filter_depth * np.sin(ph + np.pi))
        cutoff = np.clip(cutoff, 300, sr * 0.45)
        out = np.stack([_tv_lowpass(out[0], cutoff, sr), _tv_lowpass(out[1], cutoff, sr)])

    return out.astype(np.float32)


def _tv_lowpass(x: np.ndarray, cutoff: np.ndarray, sr: int, block: int = 512) -> np.ndarray:
    out = np.empty_like(x)
    zi = None
    nyq = sr / 2
    for i in range(0, len(x), block):
        fc = float(np.clip(cutoff[min(i + block // 2, len(cutoff) - 1)], 40, nyq * 0.95))
        b, a = signal.butter(2, fc / nyq)
        if zi is None:
            zi = signal.lfilter_zi(b, a) * x[0]
        out[i : i + block], zi = signal.lfilter(b, a, x[i : i + block], zi=zi)
    return out.astype(np.float32)


def incommensurate_onsets(duration_s: float, base_gap_s: float, seed: int) -> list[tuple[float, int]]:
    """Onset times for voices whose inter-onset gaps relate by irrational
    ratios (1, φ, √2, √3). Because the periods are incommensurate the voices'
    onsets essentially never coincide, so no downbeat or meter emerges — the
    listener can't parse a repeating rhythm to latch onto and predict."""
    rng = np.random.default_rng(seed)
    ratios = [(1.0, 0), (PHI, 1), (SQRT2, 2), (SQRT3, 3)]
    onsets: list[tuple[float, int]] = []
    for ratio, voice in ratios:
        t = rng.uniform(0, base_gap_s)
        while t < duration_s:
            onsets.append((t, voice))
            t += base_gap_s * ratio * rng.choice([1.0, 1 / PHI, PHI])
    onsets.sort()
    return onsets


def pink_sequence(n_notes: int, seed: int) -> np.ndarray:
    """A 1/f (pink-noise) sequence in [0, 1]. Pink processes are the signature
    of 'natural'/pleasing melodies (Voss): locally smooth, globally
    unpredictable — musical to the ear, unmodelable to the predictor."""
    rng = np.random.default_rng(seed)
    m = 1
    while m < n_notes:
        m *= 2
    white = rng.normal(size=m)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(m)
    f[0] = f[1]
    X = X / np.sqrt(f)  # 1/f amplitude shaping
    pink = np.fft.irfft(X)[:n_notes]
    pink = (pink - pink.min()) / (np.ptp(pink) + 1e-9)
    return pink


def microtonal_drift(n: int, sr: int, max_cents: float, seed: int) -> np.ndarray:
    """A slow pitch-ratio multiplier that wanders within +/- max_cents (a few
    cents = below the just-noticeable pitch difference). Keeps the auditory
    system re-evaluating the tuning it can never quite lock."""
    rng = np.random.default_rng(seed)
    n_ctrl = max(int(n / sr * 0.08) + 2, 2)
    walk = np.interp(np.arange(n), np.linspace(0, n, n_ctrl), rng.uniform(-1, 1, n_ctrl))
    cents = max_cents * walk / (np.abs(walk).max() + 1e-9)
    return 2 ** (cents / 1200)


def risset_rhythm(n: int, sr: int, base_period_s: float, layers: int = 5,
                  descend: bool = True, speed: float = 0.03) -> np.ndarray:
    """A soft-swell Risset rhythm: octave-spaced pulse layers with a sliding
    spectral window, so the pulse seems to slow (or speed) forever without ever
    arriving — the rhythmic analogue of a Shepard tone. Rendered as gentle noise
    swells rather than clicks so it stays listenable. Returns mono (n,)."""
    t = np.arange(n) / sr
    theta = (speed * t) % 1.0
    out = np.zeros(n)
    center = layers / 2
    noise = np.random.default_rng(0).normal(0, 1, n)
    b, a = signal.butter(2, [200 / (sr / 2), 1200 / (sr / 2)], btype="band")
    noise = signal.lfilter(b, a, noise)
    for k in range(layers):
        period = base_period_s * (2.0 ** k)
        rate = 1.0 / period
        pos = ((k - theta) if descend else (k + theta)) % layers
        amp = np.exp(-0.5 * ((pos - center) / (layers / 4)) ** 2)
        swell = 0.5 * (1 + np.sin(2 * np.pi * rate * t - np.pi / 2)) ** 2
        out += amp * swell * noise
    out /= np.abs(out).max() + 1e-9
    return out.astype(np.float32)

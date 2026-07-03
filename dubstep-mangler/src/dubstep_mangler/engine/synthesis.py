"""Pure-numpy synthesis: wobble bass, half-time drum kit, risers and impacts."""

from __future__ import annotations

import numpy as np
from scipy import signal

SR = 44100


def _t(dur: float, sr: int) -> np.ndarray:
    return np.arange(int(dur * sr)) / sr


def _env(n: int, sr: int, attack: float = 0.004, release: float = 0.02) -> np.ndarray:
    env = np.ones(n)
    a, r = min(int(attack * sr), n), min(int(release * sr), n)
    if a:
        env[:a] = np.linspace(0, 1, a)
    if r:
        env[-r:] *= np.linspace(1, 0, r)
    return env


def time_varying_lowpass(
    x: np.ndarray, cutoff: np.ndarray, sr: int, block: int = 256
) -> np.ndarray:
    """2nd-order lowpass whose cutoff is re-tuned every `block` samples with
    filter state carried across blocks — the classic wobble mechanism."""
    out = np.empty_like(x)
    zi = None
    nyq = sr / 2
    for i in range(0, len(x), block):
        fc = float(np.clip(cutoff[min(i + block // 2, len(cutoff) - 1)], 40, nyq * 0.95))
        b, a = signal.butter(2, fc / nyq)
        if zi is None:
            zi = signal.lfilter_zi(b, a) * x[0]
        out[i : i + block], zi = signal.lfilter(b, a, x[i : i + block], zi=zi)
    return out


def wobble_bass(
    freq: float,
    dur: float,
    sr: int = SR,
    rate_hz: float = 3.5,
    lfo_shape: str = "sine",
    drive: float = 1.8,
) -> np.ndarray:
    """LFO-swept lowpass over detuned saws + a sine sub. The dubstep growl."""
    t = _t(dur, sr)
    if not len(t):
        return np.zeros(0, dtype=np.float32)
    sub = np.sin(2 * np.pi * freq * t)
    growl = (
        signal.sawtooth(2 * np.pi * freq * 2 * t)
        + signal.sawtooth(2 * np.pi * freq * 2 * 1.007 * t)
        + 0.5 * signal.square(2 * np.pi * freq * t)
    ) / 2.5
    phase = 2 * np.pi * rate_hz * t - np.pi / 2
    if lfo_shape == "saw":
        lfo = (signal.sawtooth(phase, width=0) + 1) / 2  # falling ramp per cycle
    else:
        lfo = (np.sin(phase) + 1) / 2
    cutoff = 90 + lfo**2 * 2800 * (freq / 55.0)
    growl = time_varying_lowpass(growl, cutoff, sr)
    mix = 0.55 * sub + 0.75 * growl
    return (np.tanh(mix * drive) * _env(len(t), sr, release=0.03)).astype(np.float32)


def kick(sr: int = SR) -> np.ndarray:
    t = _t(0.35, sr)
    sweep = 160 * np.exp(-t * 22) + 44
    body = np.sin(2 * np.pi * np.cumsum(sweep) / sr) * np.exp(-t * 9)
    click = np.random.default_rng(7).normal(0, 1, len(t)) * np.exp(-t * 400)
    return (np.tanh((body + 0.4 * click) * 1.6) * 0.95).astype(np.float32)


def snare(sr: int = SR) -> np.ndarray:
    t = _t(0.3, sr)
    noise = np.random.default_rng(11).normal(0, 1, len(t))
    b, a = signal.butter(2, [1500 / (sr / 2), 9000 / (sr / 2)], btype="band")
    noise = signal.lfilter(b, a, noise) * np.exp(-t * 18)
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 30)
    return (np.tanh((noise * 1.3 + tone) * 1.4) * 0.8).astype(np.float32)


def hat(sr: int = SR, open_: bool = False) -> np.ndarray:
    dur = 0.25 if open_ else 0.06
    t = _t(dur, sr)
    noise = np.random.default_rng(13).normal(0, 1, len(t))
    b, a = signal.butter(4, 7500 / (sr / 2), btype="high")
    return (signal.lfilter(b, a, noise) * np.exp(-t * (14 if open_ else 70)) * 0.35).astype(
        np.float32
    )


def riser(dur: float, sr: int = SR) -> np.ndarray:
    """White noise swept upward + a rising detuned saw — tension into the drop."""
    t = _t(dur, sr)
    noise = np.random.default_rng(17).normal(0, 1, len(t))
    cutoff = 300 + (t / dur) ** 2 * 9000
    swept = time_varying_lowpass(noise, cutoff, sr)
    pitch = 110 * 2 ** (t / dur * 2)  # two octaves up
    saw = signal.sawtooth(2 * np.pi * np.cumsum(pitch) / sr)
    amp = (t / dur) ** 1.5
    return ((0.5 * swept + 0.25 * saw) * amp * 0.6).astype(np.float32)


def impact(sr: int = SR) -> np.ndarray:
    """Low boom + noise wash marking a drop."""
    t = _t(1.2, sr)
    boom = np.sin(2 * np.pi * (55 * np.exp(-t * 2) + 35) * t) * np.exp(-t * 3)
    noise = np.random.default_rng(19).normal(0, 1, len(t)) * np.exp(-t * 6)
    b, a = signal.butter(2, 2000 / (sr / 2))
    return (np.tanh(boom * 1.5 + signal.lfilter(b, a, noise) * 0.4) * 0.9).astype(np.float32)


def sub_drop(dur: float = 1.5, sr: int = SR) -> np.ndarray:
    """Descending sine sweep (the 'falling elevator' before a drop)."""
    t = _t(dur, sr)
    freq = 220 * np.exp(-t * 2.2) + 30
    return (np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t * 1.5) * 0.7).astype(
        np.float32
    )

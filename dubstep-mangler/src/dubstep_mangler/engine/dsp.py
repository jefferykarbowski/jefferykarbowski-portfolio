"""Reusable effects for the hypnagogic arranger: algorithmic reverb, stereo
delay, Haas widening, Shepard-tone descent. Pure numpy/scipy."""

from __future__ import annotations

import numpy as np
from scipy import signal

PHI = (1 + 5**0.5) / 2


def soft_saturate(x: np.ndarray, drive: float = 1.0) -> np.ndarray:
    return np.tanh(x * drive).astype(np.float32)


def _comb(x: np.ndarray, delay: int, fb: float) -> np.ndarray:
    """Feedback comb filter H(z)=z^-D/(1-fb*z^-D), via lfilter (fast IIR)."""
    b = np.zeros(delay + 1)
    b[delay] = 1.0
    a = np.zeros(delay + 1)
    a[0], a[delay] = 1.0, -fb
    return signal.lfilter(b, a, x)


def _allpass(x: np.ndarray, delay: int, g: float = 0.5) -> np.ndarray:
    """Schroeder allpass H(z)=(-g+z^-D)/(1-g*z^-D)."""
    b = np.zeros(delay + 1)
    b[0], b[delay] = -g, 1.0
    a = np.zeros(delay + 1)
    a[0], a[delay] = 1.0, -g
    return signal.lfilter(b, a, x)


def reverb(x: np.ndarray, sr: int, room: float = 0.85, damp: float = 0.4, mix: float = 0.35) -> np.ndarray:
    """Schroeder reverb (4 combs + 2 allpasses per channel). Input (n,) or
    (2, n); returns stereo (2, n). `damp` applies a global lowpass to the wet
    tail so the reverb darkens as it decays."""
    mono = (x if x.ndim == 1 else x.mean(axis=0)).astype(np.float64)
    comb_ms = [29.7, 37.1, 41.1, 43.7]
    ap_ms = [5.0, 1.7]
    fb = 0.7 + 0.28 * room
    chans = []
    for detune in (1.0, 1.0 + 0.012):  # slight L/R detune -> stereo width
        acc = np.zeros(len(mono))
        for cm in comb_ms:
            acc += _comb(mono, max(int(cm * detune / 1000 * sr), 1), fb)
        acc /= len(comb_ms)
        for am in ap_ms:
            acc = _allpass(acc, max(int(am * detune / 1000 * sr), 1), 0.5)
        if damp > 0:  # darken the tail
            bl, al = signal.butter(1, max(1 - damp, 0.05) * 8000 / (sr / 2))
            acc = signal.lfilter(bl, al, acc)
        chans.append(acc)
    wet = np.stack(chans)
    dry = np.stack([mono, mono])
    out = dry * (1 - mix) + wet * mix
    peak = np.abs(out).max() or 1.0
    return (out / max(peak, 1.0)).astype(np.float32)


def stereo_delay(x: np.ndarray, sr: int, time_s: float = 0.375, fb: float = 0.4, mix: float = 0.3) -> np.ndarray:
    """Ping-pong delay. Input (2, n) or (n,); returns (2, n)."""
    st = x if x.ndim == 2 else np.stack([x, x])
    d = max(int(time_s * sr), 1)
    out = st.astype(np.float64).copy()
    tap = np.zeros_like(out)
    tap[0, d:] = out[1, :-d]  # cross-feed for ping-pong
    tap[1, d:] = out[0, :-d]
    echo = tap.copy()
    for _ in range(6):
        shifted = np.zeros_like(echo)
        shifted[0, d:] = echo[1, :-d] * fb
        shifted[1, d:] = echo[0, :-d] * fb
        out += shifted * mix
        echo = shifted
    peak = np.abs(out).max() or 1.0
    return (out / max(peak, 1.0)).astype(np.float32)


def haas_widen(x: np.ndarray, sr: int, ms: float = 12.0) -> np.ndarray:
    """Widen mono to stereo via a small inter-channel delay (Haas effect)."""
    mono = x if x.ndim == 1 else x.mean(axis=0)
    d = int(ms / 1000 * sr)
    r = np.concatenate([np.zeros(d), mono])[: len(mono)]
    return np.stack([mono, r]).astype(np.float32)


def shepard_descent(n: int, sr: int, base_hz: float = 55.0, octaves: int = 6, speed: float = 0.05, seed: int = 0) -> np.ndarray:
    """An endlessly-descending Shepard glissando: `octaves` sine partials spaced
    an octave apart, all sliding downward together while a raised-cosine
    spectral window fades in new partials at the top and out at the bottom, so
    the pitch seems to fall forever without actually going anywhere. Deeply
    entrancing / mildly disorienting — a texture with no natural analogue."""
    t = np.arange(n) / sr
    theta = (speed * t) % 1.0  # 0..1 descent phase
    out = np.zeros(n)
    for k in range(octaves):
        # each partial's log-frequency descends by one octave per cycle
        logf = np.log2(base_hz) + ((k - theta) % octaves)
        freq = 2.0 ** logf
        phase = 2 * np.pi * np.cumsum(freq) / sr
        # bell-shaped amplitude over the octave range (quiet at extremes)
        center = octaves / 2
        pos = (k - theta) % octaves
        amp = np.exp(-0.5 * ((pos - center) / (octaves / 4)) ** 2)
        out += amp * np.sin(phase)
    out /= np.abs(out).max() + 1e-9
    return out.astype(np.float32)

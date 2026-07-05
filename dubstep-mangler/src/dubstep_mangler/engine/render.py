"""Mixdown: sidechain pump, bus saturation, normalization, file output."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from .arrange import Arrangement


def _sidechain_env(n: int, sr: int, kick_times: list[float], depth: float = 0.55) -> np.ndarray:
    """Gain envelope that ducks at every kick and recovers over ~250ms —
    the pumping glue between kick and bass."""
    env = np.ones(n, dtype=np.float32)
    rec = int(0.25 * sr)
    curve = 1.0 - depth * (1.0 - np.linspace(0, 1, rec) ** 0.6).astype(np.float32)
    for t in kick_times:
        i = int(t * sr)
        if 0 <= i < n:
            seg = curve[: n - i]
            env[i : i + len(seg)] = np.minimum(env[i : i + len(seg)], seg)
    return env


def mixdown(arr: Arrangement) -> np.ndarray:
    n = max(len(v) for v in arr.stems.values())
    duck = _sidechain_env(n, arr.sr, arr.kick_times)
    gains = {"drums": 1.0, "bass": 0.95, "music": 0.9, "fx": 0.85}
    mix = np.zeros(n, dtype=np.float32)
    for name, buf in arr.stems.items():
        x = np.pad(buf, (0, n - len(buf)))
        if name in ("bass", "music"):
            x = x * duck
        mix += x * gains.get(name, 1.0)
    mix = np.tanh(mix * 1.1)  # bus saturation / safety limiter
    peak = float(np.abs(mix).max()) or 1.0
    return (mix * (10 ** (-1.0 / 20) / peak)).astype(np.float32)


def write(path: str, y: np.ndarray, sr: int) -> str:
    sf.write(path, y, sr, subtype="PCM_16")
    return path

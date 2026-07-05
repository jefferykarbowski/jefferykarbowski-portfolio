"""Synthesize a small test 'song' (chords, bassline, drums, melody) so the
pipeline can be exercised end-to-end without copyrighted material."""

from __future__ import annotations

import sys

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100
BPM = 120
BEAT = 60 / BPM


def note(freq: float, dur: float, wave: str = "saw", gain: float = 0.3) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    osc = signal.sawtooth(2 * np.pi * freq * t) if wave == "saw" else np.sin(2 * np.pi * freq * t)
    env = np.ones_like(t)
    a, r = int(0.01 * SR), int(0.05 * SR)
    env[:a] = np.linspace(0, 1, a)
    env[-r:] *= np.linspace(1, 0, r)
    return (osc * env * gain).astype(np.float32)


def midi(m: float) -> float:
    return 440 * 2 ** ((m - 69) / 12)


def main(out_path: str) -> None:
    bars = 8
    n = int(bars * 4 * BEAT * SR) + SR
    mix = np.zeros(n, dtype=np.float32)

    def add(t, x):
        i = int(t * SR)
        x = x[: n - i]
        mix[i : i + len(x)] += x

    # Am F C G progression, one chord per bar
    chords = [[57, 60, 64], [53, 57, 60], [48, 52, 55], [55, 59, 62]]
    lead = [69, 72, 76, 74, 72, 69, 67, 69]
    rng = np.random.default_rng(42)
    for b in range(bars):
        t0 = b * 4 * BEAT
        for m in chords[b % 4]:
            add(t0, note(midi(m), 4 * BEAT, "saw", 0.12))
        add(t0, note(midi(chords[b % 4][0] - 24), 4 * BEAT, "sine", 0.35))  # bass
        for beat in range(4):
            add(t0 + beat * BEAT, note(midi(lead[(b * 4 + beat) % 8]), BEAT * 0.9, "saw", 0.15))
        # drums: noisy kick/snare/hat approximations
        for beat in range(4):
            tt = np.arange(int(0.2 * SR)) / SR
            kick = np.sin(2 * np.pi * (120 * np.exp(-tt * 25) + 50) * tt) * np.exp(-tt * 12)
            add(t0 + beat * BEAT, (kick * 0.8).astype(np.float32))
            if beat in (1, 3):
                sn = rng.normal(0, 1, int(0.15 * SR)) * np.exp(-np.arange(int(0.15 * SR)) / SR * 25)
                add(t0 + beat * BEAT, (sn * 0.25).astype(np.float32))
            for half in range(2):
                ht = rng.normal(0, 1, int(0.04 * SR)) * np.exp(-np.arange(int(0.04 * SR)) / SR * 80)
                add(t0 + beat * BEAT + half * BEAT / 2, (ht * 0.1).astype(np.float32))

    mix = np.tanh(mix)
    sf.write(out_path, mix, SR, subtype="PCM_16")
    print(f"wrote {out_path} ({len(mix)/SR:.1f}s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_song.wav")

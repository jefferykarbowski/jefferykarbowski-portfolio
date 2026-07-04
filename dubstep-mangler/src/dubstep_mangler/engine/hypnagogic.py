"""The hypnagogic weaver: reassembles a song's material into slow, evolving,
genreless music designed to draw the listener toward the hypnagogic (sleep-
onset) state as gently and quickly as the iso principle allows.

Design, grounded in the entrainment/sleep-onset literature:
  * No drops, no hard transients — the arousal curve only ever descends.
  * Tempo starts near a relaxed body rhythm and slows across the piece.
  * Consonant modal/pentatonic melody in the source's own key (melody and
    harmony measurably aid theta-band induction).
  * Long reverb tails and slow crossfades dissolve rhythmic edges.
  * A near-inaudible Shepard-tone glide falls forever underneath, and a soft
    heartbeat pulse decelerates from ~66 to ~50 BPM to pace the body down.

The binaural/isochronic entrainment layer is added separately (see the
entrainment module) so the beat can be set truly subliminal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy import signal

from . import dsp
from .analysis import SongAnalysis

PHI = (1 + 5**0.5) / 2

# Scale degrees (semitones) for calm, consonant modes
SCALES = {
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
}


@dataclass
class WeaveParams:
    duration_s: float = 90.0
    seed: int = 0
    scale: str = "pentatonic_minor"
    pulse_start_bpm: float = 66.0
    pulse_end_bpm: float = 50.0
    depth: float = 1.0  # overall lushness (reverb/shimmer amount)


@dataclass
class Weave:
    audio: np.ndarray  # (2, n)
    sr: int
    sections: list[tuple[str, float, float]] = field(default_factory=list)


def _pad_from_slice(seg: np.ndarray, target: int, sr: int) -> np.ndarray:
    """Stretch a slice into a smooth evolving pad: heavy time-stretch, octave
    stack, lowpass, long fades."""
    if len(seg) < 2048:
        seg = np.tile(seg, int(np.ceil(2048 / max(len(seg), 1))))
    rate = float(np.clip(len(seg) / target, 0.03, 1.0))
    try:
        stretched = librosa.effects.time_stretch(seg.astype(np.float64), rate=rate)
    except Exception:
        stretched = seg.astype(np.float64)
    stretched = np.resize(stretched, target)
    # octave-below shimmer for body
    low = signal.resample(stretched, target * 2)[:target] if target > 4 else stretched
    pad = stretched + 0.5 * low
    b, a = signal.butter(2, 1200 / (sr / 2))
    pad = signal.lfilter(b, a, pad)
    f = min(int(0.4 * sr), target // 2)
    if f:
        pad[:f] *= np.linspace(0, 1, f)
        pad[-f:] *= np.linspace(1, 0, f)
    pad /= np.abs(pad).max() + 1e-9
    return pad.astype(np.float32)


def _bell(freq: float, dur: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """A soft FM bell/pluck — the melodic voice. Gentle attack, long decay."""
    t = np.arange(int(dur * sr)) / sr
    if not len(t):
        return np.zeros(0, dtype=np.float32)
    mod = np.sin(2 * np.pi * freq * PHI * t) * np.exp(-t * 6) * 2.5
    tone = np.sin(2 * np.pi * freq * t + mod)
    tone += 0.5 * np.sin(2 * np.pi * freq * 2 * t) * np.exp(-t * 4)  # shimmer partial
    env = np.exp(-t * 2.2) * (1 - np.exp(-t * 40))  # soft attack, long tail
    return (tone * env * 0.5).astype(np.float32)


def _heartbeat(freq: float, sr: int) -> np.ndarray:
    """A soft sub thump — felt tempo anchor, not a drum."""
    t = np.arange(int(0.5 * sr)) / sr
    body = np.sin(2 * np.pi * (freq * 1.6 * np.exp(-t * 12) + freq) * t)
    return (body * np.exp(-t * 7) * 0.6).astype(np.float32)


def _midi_to_hz(m: float) -> float:
    return float(440.0 * 2 ** ((m - 69) / 12))


def weave(analysis: SongAnalysis, params: WeaveParams) -> Weave:
    sr = analysis.sr
    rng = np.random.default_rng(params.seed)
    n = int(params.duration_s * sr)
    y = analysis.y

    L = np.zeros(n, dtype=np.float64)
    R = np.zeros(n, dtype=np.float64)

    def add_st(buf_l, buf_r, t, xl, xr, g=1.0):
        i = int(t * sr)
        if i >= n or i < 0:
            return
        m = min(len(xl), n - i)
        buf_l[i : i + m] += xl[:m] * g
        buf_r[i : i + m] += xr[:m] * g

    # --- evolving pad bed from the most tonal slices ----------------------
    tonal = sorted(
        [s for s in analysis.slices if s.flatness < 0.15 and s.n_samples > int(0.2 * sr)],
        key=lambda s: s.rms,
        reverse=True,
    )[:24] or analysis.slices[:8]

    pad_len = int(PHI * 6 * sr)  # ~ long overlapping blooms
    hop = pad_len // 2
    t = 0.0
    while t * sr < n:
        s = tonal[rng.integers(len(tonal))]
        pad = _pad_from_slice(s.samples(y), pad_len, sr)
        wide = dsp.haas_widen(pad, sr, ms=rng.uniform(8, 18))
        wet = dsp.reverb(wide, sr, room=0.9, damp=0.5, mix=0.45 * params.depth)
        add_st(L, R, t, wet[0], wet[1], g=0.5)
        t += hop / sr

    # --- generative modal melody in the detected key ----------------------
    scale = SCALES.get(params.scale, SCALES["pentatonic_minor"])
    root_midi = 60 + analysis.key_root  # around middle C
    # golden-ratio phrase lengths (beats), Markov-ish gentle contour
    beat0 = 60.0 / params.pulse_start_bpm
    degree = 0
    melody_dry = np.zeros(n, dtype=np.float64)
    t = 4 * beat0  # let the pad breathe first
    while t * sr < n * 0.98:
        # random walk over scale degrees, biased toward small consonant steps
        step = rng.choice([-2, -1, 0, 1, 2], p=[0.15, 0.3, 0.1, 0.3, 0.15])
        degree = int(np.clip(degree + step, -3, len(scale) + 2))
        octave = 12 * (degree // len(scale))
        midi = root_midi + scale[degree % len(scale)] + octave
        dur = rng.choice([2, 3, 5]) * beat0  # Fibonacci-ish note lengths
        note = _bell(_midi_to_hz(midi), dur + 1.5, sr, rng)
        i = int(t * sr)
        m = min(len(note), n - i)
        if m > 0:
            melody_dry[i : i + m] += note[:m]
        # slow the melodic pace toward the end (deceleration into sleep)
        prog = t * sr / n
        t += dur * (1 + prog) + rng.uniform(0, beat0)

    mel = dsp.stereo_delay(melody_dry, sr, time_s=beat0 * 1.5, fb=0.45, mix=0.4 * params.depth)
    mel = dsp.reverb(mel, sr, room=0.85, damp=0.35, mix=0.4 * params.depth)
    L += mel[0] * 0.5
    R += mel[1] * 0.5

    # --- decelerating heartbeat pulse -------------------------------------
    bass_midi = 24 + analysis.key_root
    hb = _heartbeat(_midi_to_hz(bass_midi + 12), sr)
    t = 0.0
    while t * sr < n:
        prog = t * sr / n
        bpm = params.pulse_start_bpm + (params.pulse_end_bpm - params.pulse_start_bpm) * prog
        add_st(L, R, t, hb, hb, g=0.5)
        t += 60.0 / bpm

    # --- Shepard-tone eternal descent (quiet, underneath everything) ------
    shep = dsp.shepard_descent(n, sr, base_hz=_midi_to_hz(bass_midi), octaves=6,
                               speed=0.04, seed=params.seed)
    b, a = signal.butter(2, 2500 / (sr / 2))
    shep = signal.lfilter(b, a, shep)
    L += shep * 0.12
    R += shep * 0.12

    # --- master: gentle bus, normalize to a calm -3 dBFS ------------------
    mix = np.stack([L, R])
    mix = dsp.soft_saturate(mix * 0.9, drive=1.0)
    peak = np.abs(mix).max() or 1.0
    mix = (mix * (10 ** (-3 / 20) / peak)).astype(np.float32)

    sections = [
        (name, round(a * params.duration_s, 1), round(b * params.duration_s, 1))
        for name, a, b in [("drift", 0, 0.3), ("bloom", 0.3, 0.6),
                           ("deepen", 0.6, 0.85), ("dissolve", 0.85, 1.0)]
    ]
    return Weave(audio=mix, sr=sr, sections=sections)

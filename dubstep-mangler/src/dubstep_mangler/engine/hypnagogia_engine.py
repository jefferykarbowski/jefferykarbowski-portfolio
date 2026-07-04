"""Hypnagogia Engine — an adaptive generative engine whose sole job is to reach
the hypnagogic state fast, while staying genuinely listenable.

How it differs from the Hypnagogic Weave:
  * No exposed heartbeat or naked tone. The entrainment is *baked into the music*
    — loudness, stereo position and timbre all breathe together at a descending
    alpha->theta rate (see psychoacoustic.embedded_entrainment). You feel a pulse
    you can't point to.
  * The material is deliberately un-modelable: a harmonic bed modulated by
    incommensurate (φ/√2/√3) slow LFOs so it never loops, a 1/f fractal melody,
    microtonal drift below the just-noticeable threshold, and a Risset eternal
    descent. Nothing repeats, so the cortex never builds a predictor to skip.
  * `source_blend` weaves stretched fragments of the uploaded song into the bed,
    so it stays recognizably *yours* while dissolving into the texture.

It adapts to the input: tonal centre from the detected key, initial brightness
from the source's spectral centroid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy import signal

from . import dsp
from . import psychoacoustic as psy
from .analysis import SongAnalysis

PHI = psy.PHI
SCALES = {
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "hirajoshi": [0, 2, 3, 7, 8],
}


@dataclass
class EngineParams:
    duration_s: float = 120.0
    seed: int = 0
    scale: str = "pentatonic_minor"
    entrain_start_hz: float = 10.5  # alpha
    entrain_end_hz: float = 4.5  # theta -> edge of delta
    subtlety: float = 1.0  # scales embedded-modulation depths (1 = default, <1 deeper/bolder)
    source_blend: float = 0.4  # 0 = pure synthesis, 1 = source pads prominent
    depth: float = 1.0  # reverb/space lushness
    drift_hz: float = 0.5


@dataclass
class EngineResult:
    audio: np.ndarray  # (2, n)
    sr: int
    rate_hz_start: float
    rate_hz_end: float
    sections: list[tuple[str, float, float]] = field(default_factory=list)


def _midi_hz(m: float) -> float:
    return float(440.0 * 2 ** ((m - 69) / 12))


def _drone(freq: float, n: int, sr: int, detune_cents: np.ndarray, lfo_rate: float, seed: int) -> np.ndarray:
    """A slowly-evolving drone voice: two detuned sines an octave apart, amplitude
    breathing at an (incommensurate) sub-hertz rate, with microtonal drift."""
    t = np.arange(n) / sr
    ratio = detune_cents  # microtonal multiplier array
    ph1 = 2 * np.pi * np.cumsum(freq * ratio) / sr
    ph2 = 2 * np.pi * np.cumsum(freq * 2 * ratio * (1 + 0.0009)) / sr
    rng = np.random.default_rng(seed)
    am = 0.6 + 0.4 * np.sin(2 * np.pi * lfo_rate * t + rng.uniform(0, 2 * np.pi))
    voice = (np.sin(ph1) + 0.5 * np.sin(ph2)) * am
    return voice.astype(np.float32)


def _bell(freq: float, dur: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    if not len(t):
        return np.zeros(0, dtype=np.float32)
    mod = np.sin(2 * np.pi * freq * PHI * t) * np.exp(-t * 5) * 2.0
    tone = np.sin(2 * np.pi * freq * t + mod) + 0.4 * np.sin(2 * np.pi * freq * 2 * t) * np.exp(-t * 3)
    env = np.exp(-t * 1.8) * (1 - np.exp(-t * 35))
    return (tone * env * 0.5).astype(np.float32)


def _pad_from_slice(seg: np.ndarray, target: int, sr: int) -> np.ndarray:
    if len(seg) < 2048:
        seg = np.tile(seg, int(np.ceil(2048 / max(len(seg), 1))))
    rate = float(np.clip(len(seg) / target, 0.03, 1.0))
    try:
        s = librosa.effects.time_stretch(seg.astype(np.float64), rate=rate)
    except Exception:
        s = seg.astype(np.float64)
    s = np.resize(s, target)
    b, a = signal.butter(2, 1400 / (sr / 2))
    s = signal.lfilter(b, a, s)
    f = min(int(0.5 * sr), target // 2)
    if f:
        s[:f] *= np.linspace(0, 1, f)
        s[-f:] *= np.linspace(1, 0, f)
    return (s / (np.abs(s).max() + 1e-9)).astype(np.float32)


def drive(analysis: SongAnalysis, params: EngineParams) -> EngineResult:
    sr = analysis.sr
    n = int(params.duration_s * sr)
    rng = np.random.default_rng(params.seed)
    y = analysis.y
    scale = SCALES.get(params.scale, SCALES["pentatonic_minor"])
    root_midi = 48 + analysis.key_root  # low root for the drone bed

    L = np.zeros(n, dtype=np.float64)
    R = np.zeros(n, dtype=np.float64)

    # --- harmonic drone bed: 3 voices at scale tones, incommensurate LFOs -----
    # Amplitude-LFO rates in irrational ratios so the bed never repeats.
    lfo_rates = [0.037, 0.037 * PHI, 0.037 * psy.SQRT2]
    chord_degrees = [0, 2 if len(scale) > 2 else 1, 4 if len(scale) > 4 else len(scale) - 1]
    bed = np.zeros(n, dtype=np.float64)
    for k, deg in enumerate(chord_degrees):
        midi = root_midi + scale[deg % len(scale)] + 12 * (k == 2)
        drift = psy.microtonal_drift(n, sr, max_cents=4.0, seed=params.seed + k)
        bed += _drone(_midi_hz(midi), n, sr, drift, lfo_rates[k], params.seed + 10 + k)
    bed /= np.abs(bed).max() + 1e-9

    # --- source blend: recognizable stretched fragments woven into the bed ----
    if params.source_blend > 0:
        tonal = sorted(
            [s for s in analysis.slices if s.n_samples > int(0.2 * sr)],
            key=lambda s: s.rms, reverse=True,
        )[:16] or analysis.slices[:6]
        pad_len = int(PHI * 5 * sr)
        t = 0.0
        src = np.zeros(n, dtype=np.float64)
        while t * sr < n:
            s = tonal[rng.integers(len(tonal))]
            pad = _pad_from_slice(s.samples(y), pad_len, sr)
            i = int(t * sr)
            m = min(len(pad), n - i)
            src[i : i + m] += pad[:m]
            t += pad_len / 2 / sr
        src /= np.abs(src).max() + 1e-9
        bed = bed * (1 - 0.5 * params.source_blend) + src * params.source_blend

    # --- 1/f fractal melody over incommensurate onsets ------------------------
    onsets = psy.incommensurate_onsets(params.duration_s * 0.98, base_gap_s=1.6, seed=params.seed)
    pinks = psy.pink_sequence(len(onsets) + 1, seed=params.seed)
    melody = np.zeros(n, dtype=np.float64)
    span = len(scale) * 2  # two octaves of the scale
    for idx, (t, voice) in enumerate(onsets):
        degree = int(pinks[idx] * span)
        octave = 12 * (degree // len(scale))
        midi = root_midi + 12 + scale[degree % len(scale)] + octave - 12 * (voice % 2)
        dur = rng.choice([2.0, 3.0, 5.0]) * (1.0 + 0.4 * (t / params.duration_s))
        note = _bell(_midi_hz(midi), dur, sr, rng)
        i = int(t * sr)
        m = min(len(note), n - i)
        if m > 0:
            melody[i : i + m] += note[:m] * (0.7 if voice else 1.0)
    melody /= np.abs(melody).max() + 1e-9

    # --- Risset eternal-descent shimmer (quiet, un-resolvable) ----------------
    risset = psy.risset_rhythm(n, sr, base_period_s=0.5, layers=5, descend=True, speed=0.025)
    shep = dsp.shepard_descent(n, sr, base_hz=_midi_hz(root_midi), octaves=6, speed=0.035, seed=params.seed)
    b, a = signal.butter(2, 2200 / (sr / 2))
    shep = signal.lfilter(b, a, shep)

    # --- combine dry, spatialize -------------------------------------------
    bed_st = dsp.haas_widen(bed, sr, ms=14)
    mel_st = dsp.stereo_delay(melody, sr, time_s=0.66, fb=0.42, mix=0.4 * params.depth)
    dry = np.stack([
        bed_st[0] * 0.6 + mel_st[0] * 0.5 + risset * 0.12 + shep * 0.10,
        bed_st[1] * 0.6 + mel_st[1] * 0.5 + risset * 0.12 + shep * 0.10,
    ])

    # --- EMBED the entrainment into the music's motion ------------------------
    rate = psy.rate_curve(n, sr, params.entrain_start_hz, params.entrain_end_hz,
                          params.drift_hz, params.seed)
    s = params.subtlety
    driven = psy.embedded_entrainment(
        dry, sr, rate,
        tremolo_depth=0.10 * s, pan_depth=0.35 * s, filter_depth=0.22 * s,
        seed=params.seed,
    )

    # --- lush reverb + gentle master --------------------------------------
    wet = dsp.reverb(driven, sr, room=0.9, damp=0.45, mix=0.4 * params.depth)
    mix = dsp.soft_saturate(wet * 0.9, drive=1.0)
    peak = np.abs(mix).max() or 1.0
    mix = (mix * (10 ** (-3 / 20) / peak)).astype(np.float32)

    sections = [
        (name, round(a * params.duration_s, 1), round(b * params.duration_s, 1))
        for name, a, b in [("settle (alpha)", 0, 0.25), ("drift", 0.25, 0.55),
                           ("submerge (theta)", 0.55, 0.85), ("dissolve", 0.85, 1.0)]
    ]
    return EngineResult(
        audio=mix, sr=sr,
        rate_hz_start=params.entrain_start_hz, rate_hz_end=params.entrain_end_hz,
        sections=sections,
    )

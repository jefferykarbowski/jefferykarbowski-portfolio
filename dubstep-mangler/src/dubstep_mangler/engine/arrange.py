"""The arranger: rebuilds sliced source material into a dubstep structure.

Structure comes from golden-ratio section splits; rhythm placement from
Euclidean patterns; slice sequencing from a Markov walk in timbre space;
buildup density from the Fibonacci ladder; wobble rates locked to tempo
multiples. Everything is deterministic for a given seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy import signal

from . import patterns as pat
from . import synthesis as syn
from .analysis import SongAnalysis

STEPS_PER_BAR = 16


@dataclass
class ArrangeParams:
    tempo: float = 140.0
    total_bars: int = 48
    seed: int = 0
    intensity: float = 1.0  # 0.5 = restrained, 2.0 = unhinged
    wobble_shape: str = "sine"  # "sine" | "saw"


@dataclass
class Arrangement:
    stems: dict[str, np.ndarray]
    sr: int
    tempo: float
    sections: list[tuple[str, int, int]]
    kick_times: list[float] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


class _Timeline:
    def __init__(self, n: int, sr: int):
        self.buf = np.zeros(n, dtype=np.float32)
        self.sr = sr

    def add(self, t: float, x: np.ndarray, gain: float = 1.0) -> None:
        i = int(t * self.sr)
        if i < 0 or i >= len(self.buf) or len(x) == 0:
            return
        x = x[: len(self.buf) - i]
        self.buf[i : i + len(x)] += x * gain


def _fit(seg: np.ndarray, target: int, sr: int, stretch: bool = False) -> np.ndarray:
    """Force a slice to exactly `target` samples (optionally time-stretched), with
    short fades so cut points never click."""
    if target <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(seg) == 0:
        return np.zeros(target, dtype=np.float32)
    if stretch and len(seg) >= 2048:
        rate = float(np.clip(len(seg) / target, 0.3, 3.0))
        seg = librosa.effects.time_stretch(seg.astype(np.float64), rate=rate)
    out = seg[:target].copy() if len(seg) >= target else np.pad(seg, (0, target - len(seg)))
    f = max(min(int(0.005 * sr), target // 4), 1)
    out[:f] *= np.linspace(0, 1, f)
    out[-f:] *= np.linspace(1, 0, f)
    return out.astype(np.float32)


def _highpass(x: np.ndarray, fc: float, sr: int) -> np.ndarray:
    b, a = signal.butter(2, fc / (sr / 2), btype="high")
    return signal.lfilter(b, a, x).astype(np.float32)


def _lowpass(x: np.ndarray, fc: float, sr: int) -> np.ndarray:
    b, a = signal.butter(2, fc / (sr / 2))
    return signal.lfilter(b, a, x).astype(np.float32)


def _bass_freq(pitch_class: int, semitone_offset: int) -> float:
    midi = 24 + (pitch_class + semitone_offset) % 12  # C1..B1 octave
    return float(440.0 * 2 ** ((midi - 69) / 12))


def build(analysis: SongAnalysis, params: ArrangeParams) -> Arrangement:
    sr = analysis.sr
    rng = np.random.default_rng(params.seed)
    spb = 60.0 / params.tempo  # seconds per beat
    bar = 4 * spb
    step = bar / STEPS_PER_BAR
    total_s = params.total_bars * bar + 3.0  # tail room for reverb-length decays
    n = int(total_s * sr)

    stems = {name: _Timeline(n, sr) for name in ("drums", "bass", "music", "fx")}
    sections = pat.golden_structure(params.total_bars)
    kick_times: list[float] = []
    events: list[dict] = []

    # --- slice pools -------------------------------------------------------
    y = analysis.y
    sl = analysis.slices
    if not sl:
        raise ValueError("No usable slices found in source audio")
    by_punch = sorted(sl, key=lambda s: s.onset_strength * s.rms, reverse=True)
    stabs = [s for s in by_punch if s.n_samples <= int(0.8 * sr)][:32] or sl[:32]
    tonal = sorted(
        [s for s in sl if s.flatness < 0.1 and s.n_samples >= int(0.25 * sr)],
        key=lambda s: s.rms,
        reverse=True,
    )[:16] or sl[:8]

    X = analysis.feature_matrix()
    stab_walk = pat.markov_walk(
        X[[s.index for s in stabs]],
        n_steps=params.total_bars * STEPS_PER_BAR,
        rng=rng,
        temperature=1.0 / max(params.intensity, 0.25),
        start=0,
    )
    walk_pos = 0

    def next_stab():
        nonlocal walk_pos
        s = stabs[stab_walk[walk_pos % len(stab_walk)]]
        walk_pos += 1
        return s

    # --- one-shots ---------------------------------------------------------
    KICK, SNARE, HAT, OHAT = syn.kick(sr), syn.snare(sr), syn.hat(sr), syn.hat(sr, open_=True)

    def put_kick(t: float, gain: float = 1.0):
        stems["drums"].add(t, KICK, gain)
        kick_times.append(t)

    # bass progression in semitones from the detected root
    degrees = [0, 0, 8, 10] if analysis.key_mode == "minor" else [0, 9, 5, 7]
    wobble_multiples = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0])

    def bar_time(b: int) -> float:
        return b * bar

    # --- section renderers -------------------------------------------------
    def render_intro(b0: int, b1: int, closing: bool = False):
        for b in range(b0, b1, 2):
            s = tonal[rng.integers(len(tonal))]
            bed = _fit(s.samples(y), int(2 * bar * sr), sr, stretch=True)
            bed = _lowpass(bed, 900, sr)
            if closing:  # outro: filter sweeps down to nothing
                cut = np.linspace(3000, 150, len(bed))
                bed = syn.time_varying_lowpass(bed, cut, sr)
            stems["music"].add(bar_time(b), bed, 0.45)
            for i, hit in enumerate(pat.euclidean_rhythm(3, STEPS_PER_BAR, rotation=b)):
                if hit:
                    stems["drums"].add(bar_time(b) + i * step, HAT, 0.25)
        if closing:
            for b in range(b0, min(b0 + (b1 - b0) // 2, b1)):
                for beat in range(4):
                    put_kick(bar_time(b) + beat * spb, 0.5 * (1 - (b - b0) / max(b1 - b0, 1)))

    def render_build(b0: int, b1: int):
        density = pat.fibonacci_density(b1 - b0, 2, 13)
        for k, b in enumerate(range(b0, b1)):
            grid = pat.euclidean_rhythm(min(int(density[k] * params.intensity) or 1, 16),
                                        STEPS_PER_BAR, rotation=b)
            for i, hit in enumerate(grid):
                if hit:
                    s = next_stab()
                    stems["music"].add(
                        bar_time(b) + i * step,
                        _highpass(_fit(s.samples(y), int(step * sr) * 2, sr), 250, sr),
                        0.5,
                    )
            if b >= b0 + (b1 - b0) // 2:
                for beat in range(4):
                    put_kick(bar_time(b) + beat * spb, 0.8)
            # snare roll that doubles in rate over the final 4 bars
            from_end = b1 - b
            if from_end <= 4:
                div = {4: 4, 3: 4, 2: 8, 1: 16}[from_end]
                for i in range(div):
                    stems["drums"].add(
                        bar_time(b) + i * bar / div, SNARE, 0.3 + 0.5 * (1 - from_end / 4)
                    )
        rise_bars = min(8, b1 - b0)
        stems["fx"].add(bar_time(b1 - rise_bars), syn.riser(rise_bars * bar, sr), 0.8)
        stems["fx"].add(bar_time(b1) - 1.2, syn.sub_drop(1.2, sr), 0.9)

    def render_drop(b0: int, b1: int, variant: int):
        stems["fx"].add(bar_time(b0), syn.impact(sr), 1.0)
        events.append({"type": "drop", "bar": b0, "time": round(bar_time(b0), 3)})
        kick_grid = pat.euclidean_rhythm(3, STEPS_PER_BAR, rotation=2 * variant)
        for b in range(b0, b1):
            # half-time drums: kick opens the bar, snare lands on beat 3
            put_kick(bar_time(b))
            for i, hit in enumerate(kick_grid):
                if hit and 0 < i < 8 and rng.random() < 0.5:
                    put_kick(bar_time(b) + i * step, 0.7)
            stems["drums"].add(bar_time(b) + 8 * step, SNARE, 1.0)
            if variant == 0:
                for i in range(0, STEPS_PER_BAR, 2):
                    stems["drums"].add(bar_time(b) + i * step, HAT, 0.3 + 0.15 * (i % 4 == 2))
                stems["drums"].add(bar_time(b) + 14 * step, OHAT, 0.35)
            else:
                for i, hit in enumerate(pat.euclidean_rhythm(11, STEPS_PER_BAR, rotation=b)):
                    if hit:
                        stems["drums"].add(bar_time(b) + i * step, HAT, 0.3)

            # wobble bass: root from the progression, rate locked to tempo
            root = degrees[(b - b0) % len(degrees)]
            freq = _bass_freq(analysis.key_root, root)
            weights = wobble_multiples ** (params.intensity + 0.5 * variant)
            rate = (params.tempo / 60.0) * float(
                rng.choice(wobble_multiples, p=weights / weights.sum())
            )
            if rng.random() < 0.3 * params.intensity:  # split-bar double wobble
                half = bar / 2 - step / 4
                for h, mult in enumerate(rng.choice(pat.FIB[:4], size=2)):
                    stems["bass"].add(
                        bar_time(b) + h * bar / 2,
                        syn.wobble_bass(freq, half, sr, rate * mult / 2,
                                        params.wobble_shape, drive=1.5 + params.intensity),
                        0.9,
                    )
            else:
                stems["bass"].add(
                    bar_time(b),
                    syn.wobble_bass(freq, bar - step / 4, sr, rate,
                                    params.wobble_shape, drive=1.5 + params.intensity),
                    0.9,
                )
            events.append({"type": "wobble", "bar": b, "freq_hz": round(freq, 1),
                           "rate_hz": round(rate, 2)})

            # timbre-walk stabs riding above the bass
            for i, hit in enumerate(
                pat.euclidean_rhythm(5 + 2 * variant, STEPS_PER_BAR, rotation=3 * b)
            ):
                if hit and i != 8:  # keep the snare pocket clear
                    s = next_stab()
                    length = int(step * sr) * (1 if rng.random() < 0.7 else 2)
                    stems["music"].add(
                        bar_time(b) + i * step,
                        _highpass(_fit(s.samples(y), length, sr), 350, sr),
                        0.4,
                    )

    def render_breakdown(b0: int, b1: int):
        for b in range(b0, b1, 2):
            s = tonal[rng.integers(len(tonal))]
            bed = _lowpass(_fit(s.samples(y), int(2 * bar * sr), sr, stretch=True), 1400, sr)
            stems["music"].add(bar_time(b), bed, 0.5)
            if (b - b0) % 4 < 2:
                put_kick(bar_time(b), 0.5)
                stems["drums"].add(bar_time(b) + 8 * step, SNARE, 0.4)
        rise = min(4, b1 - b0)
        stems["fx"].add(bar_time(b1 - rise), syn.riser(rise * bar, sr), 0.7)
        stems["fx"].add(bar_time(b1) - 1.0, syn.sub_drop(1.0, sr), 0.8)

    for name, a, b in sections:
        if name == "intro":
            render_intro(a, b)
        elif name in ("build",):
            render_build(a, b)
        elif name == "drop":
            render_drop(a, b, variant=0)
        elif name == "breakdown":
            render_breakdown(a, b)
        elif name == "drop2":
            render_drop(a, b, variant=1)
        elif name == "outro":
            render_intro(a, b, closing=True)

    return Arrangement(
        stems={k: v.buf for k, v in stems.items()},
        sr=sr,
        tempo=params.tempo,
        sections=sections,
        kick_times=sorted(kick_times),
        events=events,
    )

"""Psychedelic DJ — a two-track algorithmic DJ/producer.

Takes two songs: one is sliced for RHYTHM (its percussive hits become the beat),
the other is mined for MELODY & HARMONY (its tonal grains become pads and a
granular lead). The engine beat-matches them (detect each tempo, stretch to a
shared grid), key-matches them (detect each key, pitch-shift the melodic
material to agree), then arranges a modern psychedelic / avant-garde track WITH
A BEAT: Euclidean grooves crossed with irrational-ratio polyrhythm, glitch
stutters and reverses, a granular lead, evolving pads, sub bass, and a thread of
the embedded 'vibration' pulse kept subtle under the groove.

Not a neural net — an algorithmic DJ. But beat-matching, key-matching and
generative arrangement is what a DJ does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
from scipy import signal

from . import dsp
from . import patterns as pat
from . import psychoacoustic as psy
from .analysis import SongAnalysis
from .arrange import _fit, _highpass, _lowpass

PHI = psy.PHI
STEPS = 16  # sixteenth-note grid

SCALES = {
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "hirajoshi": [0, 2, 3, 7, 8],
}


@dataclass
class DJParams:
    duration_s: float = 90.0
    seed: int = 0
    tempo: float = 0.0  # 0 = auto from the rhythm source
    scale: str = "pentatonic_minor"
    intensity: float = 1.0  # drum density
    weirdness: float = 0.5  # avant-garde: polyrhythm, odd accents, glitch probability
    entrainment: float = 0.5  # 0 = off, 1 = full embedded pulse under the groove
    entrain_hz: float = 8.0  # steady embedded pulse rate (theta/alpha)
    depth: float = 1.0  # fx lushness


@dataclass
class DJResult:
    audio: np.ndarray  # (2, n)
    sr: int
    tempo: float
    key_name: str
    sections: list[tuple[str, int, int]] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


def _key_interval(from_root: int, to_root: int) -> int:
    d = (to_root - from_root) % 12
    return d - 12 if d > 6 else d


def _pools(a: SongAnalysis):
    sr = a.sr
    sl = a.slices
    perc = sorted(sl, key=lambda s: s.onset_strength * s.rms, reverse=True)
    kick = [s for s in perc if s.centroid < 1600][:32] or perc[:12]
    snare = [s for s in perc if 1600 <= s.centroid < 4500][:32] or perc[:12]
    hat = [s for s in perc if s.centroid >= 4500 and s.n_samples < int(0.3 * sr)][:32] or perc[:12]
    tonal = sorted(
        [s for s in sl if s.flatness < 0.18 and s.n_samples > int(0.2 * sr)],
        key=lambda s: s.rms, reverse=True,
    )[:32] or sl[:8]
    return kick, snare, hat, tonal


def _pitch(seg: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    if abs(n_steps) < 0.01 or len(seg) < 256:
        return seg.astype(np.float32)
    try:
        return librosa.effects.pitch_shift(seg.astype(np.float64), sr=sr, n_steps=n_steps).astype(np.float32)
    except Exception:
        return seg.astype(np.float32)


class _Buf:
    def __init__(self, n):
        self.b = np.zeros(n, dtype=np.float32)

    def add(self, i, x, g=1.0):
        if i < 0 or i >= len(self.b) or len(x) == 0:
            return
        x = x[: len(self.b) - i]
        self.b[i : i + len(x)] += x * g


def mix_tracks(rhythm: SongAnalysis, melody: SongAnalysis | None, params: DJParams) -> DJResult:
    sr = rhythm.sr
    mel = melody or rhythm
    rng = np.random.default_rng(params.seed)

    tempo = params.tempo or rhythm.tempo
    if not np.isfinite(tempo) or tempo <= 0:
        tempo = 120.0
    tempo = float(np.clip(tempo, 70, 150))
    spb = 60.0 / tempo
    bar = 4 * spb
    step = bar / STEPS
    total_bars = max(int(params.duration_s / bar), 8)
    n = int((total_bars * bar + 4) * sr)

    stems = {k: _Buf(n) for k in ("drums", "bass", "pad", "lead", "fx")}
    kick_p, snare_p, hat_p, r_tonal = _pools(rhythm)
    _, _, _, m_tonal = _pools(mel)
    y_r, y_m = rhythm.y, mel.y

    # key-match: shift melodic material from its key to the rhythm's key
    key_root = rhythm.key_root
    mel_shift = _key_interval(mel.key_root, key_root)
    scale = SCALES.get(params.scale, SCALES["pentatonic_minor"])
    sections = pat.golden_structure(total_bars)
    events: list[dict] = []

    def bt(b):
        return b * bar

    def place(buf, t, seg, target, g, stretch=True, hp=None, lp=None):
        x = _fit(seg, target, sr, stretch=stretch)
        if hp:
            x = _highpass(x, hp, sr)
        if lp:
            x = _lowpass(x, lp, sr)
        stems[buf].add(int(t * sr), x, g)

    kick_times: list[float] = []

    # --- sub bass (synth, follows a root motion in the matched key) ----------
    degrees = [0, 0, -2, 3] if "minor" in params.scale or params.scale == "hirajoshi" else [0, 4, 5, 3]

    def sub(freq, dur):
        t = np.arange(int(dur * sr)) / sr
        env = np.exp(-t * 1.5) * (1 - np.exp(-t * 80))
        tone = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        return (np.tanh(tone * 1.3) * env * 0.7).astype(np.float32)

    def midi_hz(m):
        return 440.0 * 2 ** ((m - 69) / 12)

    # --- per-section rendering ---------------------------------------------
    def drums_for(b, density, section):
        rot = b * 3
        kg = pat.euclidean_rhythm(max(int(density), 1), STEPS, rotation=rot)
        for i, hit in enumerate(kg):
            if hit and (i % 4 == 0 or rng.random() < 0.4 * params.intensity):
                s = kick_p[rng.integers(len(kick_p))]
                place("drums", bt(b) + i * step, s.samples(y_r), int(step * sr * 1.5), 1.0, lp=3500)
                kick_times.append(bt(b) + i * step)
        # backbeat snare, sometimes displaced (avant-garde)
        for beat in (1, 3):
            disp = step if (params.weirdness > 0.5 and rng.random() < params.weirdness * 0.4) else 0
            s = snare_p[rng.integers(len(snare_p))]
            place("drums", bt(b) + beat * spb + disp, s.samples(y_r), int(step * sr * 2), 0.9, hp=800)
        # hats
        hg = pat.euclidean_rhythm(int(7 + 6 * params.intensity), STEPS, rotation=b)
        for i, hit in enumerate(hg):
            if hit:
                s = hat_p[rng.integers(len(hat_p))]
                place("drums", bt(b) + i * step, s.samples(y_r), int(step * sr * 0.8),
                      0.35, hp=5000)
        # irrational-ratio polyrhythm layer (3-, 5-, 7-tuplets over the bar)
        if params.weirdness > 0.25:
            tup = int(rng.choice([3, 5, 7], p=[0.5, 0.3, 0.2]))
            for k in range(tup):
                if rng.random() < params.weirdness:
                    pool = snare_p if rng.random() < 0.5 else hat_p
                    s = pool[rng.integers(len(pool))]
                    place("drums", bt(b) + k * bar / tup, s.samples(y_r), int(step * sr),
                          0.3, hp=2000)
        # glitch stutter: retrigger one slice into fast subdivisions
        if section in ("build", "drop", "drop2") and rng.random() < 0.35 * params.weirdness:
            s = kick_p[rng.integers(len(kick_p))]
            reps = int(rng.choice([4, 6, 8]))
            start = rng.integers(STEPS - reps)
            for r in range(reps):
                seg = s.samples(y_r)
                if rng.random() < 0.5:
                    seg = seg[::-1]  # reverse glitch
                place("drums", bt(b) + (start + r * (reps / STEPS)) * step, seg,
                      int(step * sr * reps / STEPS), 0.5, stretch=False, hp=1200)
            events.append({"type": "glitch", "bar": int(b), "reps": reps})

    def bass_for(b):
        deg = degrees[b % len(degrees)]
        freq = midi_hz(24 + (key_root + scale[deg % len(scale)]) % 12)
        stems["bass"].add(int(bt(b) * sr), sub(freq, bar * 0.95), 0.9)

    def pad_for(b):
        if b % 2:
            return
        s = m_tonal[rng.integers(len(m_tonal))]
        seg = _pitch(s.samples(y_m), sr, mel_shift)  # key-matched
        x = _fit(seg, int(2 * bar * sr), sr, stretch=True)
        stems["pad"].add(int(bt(b) * sr), _lowpass(x, 1600, sr), 0.5)

    # granular lead: melody-source grains pitch-shifted to a 1/f scale melody
    def lead_bed():
        onsets = psy.incommensurate_onsets(params.duration_s * 0.95, base_gap_s=spb * 2, seed=params.seed)
        pinks = psy.pink_sequence(len(onsets) + 1, params.seed)
        span = len(scale)
        for idx, (t, voice) in enumerate(onsets):
            if rng.random() > 0.6 + 0.3 * params.intensity:
                continue
            s = m_tonal[rng.integers(len(m_tonal))]
            grain = s.samples(y_m)[: int(0.5 * sr)]
            deg = int(pinks[idx] * span)
            semis = scale[deg % len(scale)] + 12 * (deg // len(scale)) - 5 + mel_shift
            g = _pitch(grain, sr, semis)
            g = _fit(g, int(rng.choice([2, 3]) * step * sr * 2), sr, stretch=False)
            stems["lead"].add(int(t * sr), _highpass(g, 300, sr), 0.4 * (0.7 if voice else 1.0))

    density_by_section = {"intro": 3, "build": 6, "drop": 9, "breakdown": 4, "drop2": 10, "outro": 3}
    for name, a, b in sections:
        dens = density_by_section.get(name, 6) * (0.6 + 0.4 * params.intensity)
        for bar_i in range(a, b):
            if name != "breakdown":
                drums_for(bar_i, dens, name)
                bass_for(bar_i)
            pad_for(bar_i)
        if name in ("drop", "drop2"):
            stems["fx"].add(int(bt(a) * sr), dsp.shepard_descent(int(2 * bar * sr), sr,
                            base_hz=midi_hz(36 + key_root), octaves=5, speed=0.05, seed=params.seed)[: int(2*bar*sr)], 0.2)
    lead_bed()

    # --- process lead with ping-pong delay + reverb -------------------------
    lead = dsp.stereo_delay(stems["lead"].b, sr, time_s=spb * 1.5, fb=0.4, mix=0.4 * params.depth)
    lead = dsp.reverb(lead, sr, room=0.8, damp=0.3, mix=0.3 * params.depth)
    pad_st = dsp.reverb(dsp.haas_widen(stems["pad"].b, sr, ms=18), sr,
                        room=0.88, damp=0.5, mix=0.42 * params.depth)

    # --- sidechain bass + pad to the kick -----------------------------------
    duck = np.ones(n, dtype=np.float32)
    rec = int(0.18 * sr)
    curve = 1.0 - 0.5 * (1.0 - np.linspace(0, 1, rec) ** 0.6).astype(np.float32)
    for t in kick_times:
        i = int(t * sr)
        if 0 <= i < n:
            seg = curve[: n - i]
            duck[i : i + len(seg)] = np.minimum(duck[i : i + len(seg)], seg)

    L = (stems["drums"].b + stems["bass"].b * duck + pad_st[0] * duck * 0.9
         + lead[0] * 0.8 + stems["fx"].b)
    R = (stems["drums"].b + stems["bass"].b * duck + pad_st[1] * duck * 0.9
         + lead[1] * 0.8 + stems["fx"].b)
    mix = np.stack([L, R]).astype(np.float32)

    # --- optional embedded 'vibration' pulse under the groove ---------------
    if params.entrainment > 0:
        driven = psy.embedded_entrainment(mix, sr, np.full(n, params.entrain_hz),
                                          tremolo_depth=0.06 * params.entrainment,
                                          pan_depth=0.15 * params.entrainment,
                                          filter_depth=0.0, seed=params.seed)
        mix = mix * (1 - params.entrainment) + driven * params.entrainment

    mix = np.tanh(mix * 1.05)
    peak = np.abs(mix).max() or 1.0
    mix = (mix * (10 ** (-1.0 / 20) / peak)).astype(np.float32)
    # short fades
    f = min(int(0.05 * sr), n // 10)
    mix[:, :f] *= np.linspace(0, 1, f)
    mix[:, -f * 20 :] *= np.linspace(1, 0, f * 20) if f * 20 < n else 1.0

    return DJResult(audio=mix, sr=sr, tempo=tempo, key_name=rhythm.key_name,
                    sections=sections, events=events)

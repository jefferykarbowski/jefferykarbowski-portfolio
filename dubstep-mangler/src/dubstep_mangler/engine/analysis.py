"""Load a song, slice it on onset/beat boundaries, and fingerprint every slice."""

from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np

SR = 44100

# Krumhansl-Schmuckler key profiles (major, minor)
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)
_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class Slice:
    index: int
    start: int  # sample offset into the source
    end: int
    centroid: float  # mean spectral centroid (Hz) — brightness
    rms: float  # loudness
    flatness: float  # noisiness (0=tonal, 1=noise)
    onset_strength: float  # percussiveness at the slice start
    mfcc: np.ndarray = field(repr=False)  # timbre fingerprint (13,)

    @property
    def n_samples(self) -> int:
        return self.end - self.start

    def samples(self, y: np.ndarray) -> np.ndarray:
        return y[self.start : self.end]


@dataclass
class SongAnalysis:
    y: np.ndarray = field(repr=False)
    sr: int
    duration: float
    tempo: float
    key_root: int  # pitch class 0-11
    key_mode: str  # "major" | "minor"
    slices: list[Slice] = field(repr=False)

    @property
    def key_name(self) -> str:
        return f"{_PITCH_NAMES[self.key_root]} {self.key_mode}"

    def feature_matrix(self) -> np.ndarray:
        """Per-slice feature vectors used for similarity: [mfcc..., centroid, rms, flatness]."""
        rows = []
        for s in self.slices:
            rows.append(
                np.concatenate(
                    [s.mfcc, [np.log1p(s.centroid), s.rms * 10.0, s.flatness * 5.0]]
                )
            )
        X = np.array(rows)
        # z-score each dimension so no single feature dominates distances
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
        return X


def load_audio(path: str, sr: int = SR) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32), sr


def estimate_key(y: np.ndarray, sr: int) -> tuple[int, str]:
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    best = (-np.inf, 0, "minor")
    for root in range(12):
        for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            score = np.corrcoef(np.roll(profile, root), chroma)[0, 1]
            if score > best[0]:
                best = (score, root, mode)
    return best[1], best[2]


def _slice_boundaries(y: np.ndarray, sr: int, tempo_beats: np.ndarray) -> np.ndarray:
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="samples", backtrack=True)
    bounds = np.unique(np.concatenate([[0], onsets, tempo_beats, [len(y)]]))
    # merge boundaries closer than 90ms, split segments longer than 2s on an even grid
    min_len, max_len = int(0.09 * sr), int(2.0 * sr)
    merged = [int(bounds[0])]
    for b in bounds[1:]:
        if b - merged[-1] >= min_len:
            merged.append(int(b))
    merged[-1] = len(y)
    final: list[int] = []
    for a, b in zip(merged, merged[1:]):
        final.append(a)
        if b - a > max_len:
            n = int(np.ceil((b - a) / max_len))
            final.extend(int(a + i * (b - a) / n) for i in range(1, n))
    final.append(len(y))
    return np.array(sorted(set(final)))


def analyze(path: str, sr: int = SR) -> SongAnalysis:
    y, sr = load_audio(path, sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units="samples")
    tempo = float(np.atleast_1d(tempo)[0]) or 120.0
    root, mode = estimate_key(y, sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    bounds = _slice_boundaries(y, sr, beats)

    slices: list[Slice] = []
    hop = 512
    for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        seg = y[a:b]
        if len(seg) < hop:
            continue
        n_fft = min(2048, len(seg))
        centroid = librosa.feature.spectral_centroid(y=seg, sr=sr, n_fft=n_fft).mean()
        flatness = librosa.feature.spectral_flatness(y=seg, n_fft=n_fft).mean()
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13, n_fft=n_fft).mean(axis=1)
        frame = min(a // hop, len(onset_env) - 1)
        slices.append(
            Slice(
                index=len(slices),
                start=int(a),
                end=int(b),
                centroid=float(centroid),
                rms=float(np.sqrt(np.mean(seg**2))),
                flatness=float(flatness),
                onset_strength=float(onset_env[frame]),
                mfcc=mfcc.astype(np.float64),
            )
        )

    return SongAnalysis(
        y=y,
        sr=sr,
        duration=len(y) / sr,
        tempo=tempo,
        key_root=root,
        key_mode=mode,
        slices=slices,
    )

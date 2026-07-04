"""High-level pipeline: analyze -> arrange -> render. Shared by CLI and MCP server."""

from __future__ import annotations

from pathlib import Path

from .engine import analysis, arrange, dj, hypnagogic, hypnagogia_engine, render


def analyze_summary(path: str) -> dict:
    a = analysis.analyze(path)
    slices = a.slices
    return {
        "path": str(path),
        "duration_s": round(a.duration, 2),
        "detected_tempo_bpm": round(a.tempo, 1),
        "detected_key": a.key_name,
        "num_slices": len(slices),
        "slice_stats": {
            "mean_length_ms": round(
                1000 * sum(s.n_samples for s in slices) / (len(slices) * a.sr), 1
            ),
            "brightest_centroid_hz": round(max(s.centroid for s in slices), 0),
            "most_percussive_index": max(
                slices, key=lambda s: s.onset_strength
            ).index,
        },
    }


def mangle_audio(
    y,
    sr: int,
    tempo: float = 140.0,
    total_bars: int = 48,
    seed: int = 0,
    intensity: float = 1.0,
    wobble_shape: str = "sine",
):
    """In-memory variant used by the ComfyUI nodes: mono array in,
    (mix array, sample_rate, metadata) out."""
    a = analysis.analyze_audio(y, sr)
    params = arrange.ArrangeParams(
        tempo=tempo,
        total_bars=total_bars,
        seed=seed,
        intensity=intensity,
        wobble_shape=wobble_shape,
    )
    arrangement = arrange.build(a, params)
    mix = render.mixdown(arrangement)
    meta = {
        "duration_s": round(len(mix) / arrangement.sr, 2),
        "source": {
            "detected_tempo_bpm": round(a.tempo, 1),
            "detected_key": a.key_name,
            "num_slices": len(a.slices),
        },
        "structure": [
            {"section": name, "bars": f"{s}-{e}"} for name, s, e in arrangement.sections
        ],
        "drops": [e for e in arrangement.events if e["type"] == "drop"],
        "seed": seed,
    }
    return mix, arrangement.sr, meta


def weave_audio(
    y,
    sr: int,
    duration_s: float = 90.0,
    seed: int = 0,
    scale: str = "pentatonic_minor",
    pulse_start_bpm: float = 66.0,
    pulse_end_bpm: float = 50.0,
    depth: float = 1.0,
):
    """In-memory hypnagogic weave: mono array in, (stereo (2,n), sr, meta) out."""
    a = analysis.analyze_audio(y, sr)
    params = hypnagogic.WeaveParams(
        duration_s=duration_s,
        seed=seed,
        scale=scale,
        pulse_start_bpm=pulse_start_bpm,
        pulse_end_bpm=pulse_end_bpm,
        depth=depth,
    )
    w = hypnagogic.weave(a, params)
    meta = {
        "duration_s": round(w.audio.shape[-1] / w.sr, 2),
        "source": {"detected_key": a.key_name, "num_slices": len(a.slices)},
        "scale": scale,
        "sections": [{"section": nm, "from_s": s, "to_s": e} for nm, s, e in w.sections],
        "seed": seed,
    }
    return w.audio, w.sr, meta


def dj_audio(
    rhythm_y,
    melody_y,
    sr: int,
    duration_s: float = 90.0,
    seed: int = 0,
    tempo: float = 0.0,
    scale: str = "pentatonic_minor",
    intensity: float = 1.0,
    weirdness: float = 0.5,
    entrainment: float = 0.5,
    entrain_hz: float = 8.0,
    depth: float = 1.0,
):
    """Two-track Psychedelic DJ: rhythm source + melody source -> avant-garde
    track with a beat. melody_y may be None (rhythm source used for both)."""
    r = analysis.analyze_audio(rhythm_y, sr)
    m = analysis.analyze_audio(melody_y, sr) if melody_y is not None else None
    params = dj.DJParams(
        duration_s=duration_s, seed=seed, tempo=tempo, scale=scale,
        intensity=intensity, weirdness=weirdness, entrainment=entrainment,
        entrain_hz=entrain_hz, depth=depth,
    )
    res = dj.mix_tracks(r, m, params)
    meta = {
        "duration_s": round(res.audio.shape[-1] / res.sr, 2),
        "tempo_bpm": round(res.tempo, 1),
        "matched_key": res.key_name,
        "rhythm_source": {"detected_tempo_bpm": round(r.tempo, 1), "detected_key": r.key_name},
        "melody_source": ({"detected_key": m.key_name} if m else "same as rhythm"),
        "sections": [{"section": nm, "bars": f"{s}-{e}"} for nm, s, e in res.sections],
        "glitch_events": [e for e in res.events if e["type"] == "glitch"][:8],
        "seed": seed,
    }
    return res.audio, res.sr, meta


def drive_audio(
    y,
    sr: int,
    duration_s: float = 120.0,
    seed: int = 0,
    scale: str = "pentatonic_minor",
    entrain_start_hz: float = 10.5,
    entrain_end_hz: float = 4.5,
    subtlety: float = 1.0,
    source_blend: float = 0.4,
    depth: float = 1.0,
    drift_hz: float = 0.5,
):
    """In-memory Hypnagogia Engine: mono array in, (stereo (2,n), sr, meta) out.
    The entrainment is embedded in the music's motion, not added as a tone."""
    a = analysis.analyze_audio(y, sr)
    params = hypnagogia_engine.EngineParams(
        duration_s=duration_s, seed=seed, scale=scale,
        entrain_start_hz=entrain_start_hz, entrain_end_hz=entrain_end_hz,
        subtlety=subtlety, source_blend=source_blend, depth=depth, drift_hz=drift_hz,
    )
    r = hypnagogia_engine.drive(a, params)
    meta = {
        "duration_s": round(r.audio.shape[-1] / r.sr, 2),
        "source": {"detected_key": a.key_name, "num_slices": len(a.slices)},
        "scale": scale,
        "entrainment": {
            "carrier": "embedded in loudness/pan/timbre (no exposed tone)",
            "descent_hz": f"{r.rate_hz_start} (alpha) -> {r.rate_hz_end} (theta)",
            "structure": "incommensurate phi/sqrt2 LFOs + 1/f melody + Risset descent (non-repeating)",
        },
        "sections": [{"section": nm, "from_s": s, "to_s": e} for nm, s, e in r.sections],
        "seed": seed,
    }
    return r.audio, r.sr, meta


def mangle(
    input_path: str,
    output_path: str | None = None,
    tempo: float = 140.0,
    total_bars: int = 48,
    seed: int = 0,
    intensity: float = 1.0,
    wobble_shape: str = "sine",
) -> dict:
    """Slice `input_path` apart and reassemble it as a dubstep track."""
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"No such audio file: {src}")
    if output_path is None:
        output_path = str(src.with_name(f"{src.stem}_dubstep_seed{seed}.wav"))

    y, sr = analysis.load_audio(str(src))
    a = analysis.analyze_audio(y, sr)
    params = arrange.ArrangeParams(
        tempo=tempo,
        total_bars=total_bars,
        seed=seed,
        intensity=intensity,
        wobble_shape=wobble_shape,
    )
    arrangement = arrange.build(a, params)
    mix = render.mixdown(arrangement)
    render.write(output_path, mix, arrangement.sr)

    return {
        "output_path": output_path,
        "duration_s": round(len(mix) / arrangement.sr, 2),
        "tempo_bpm": tempo,
        "source": {
            "detected_tempo_bpm": round(a.tempo, 1),
            "detected_key": a.key_name,
            "num_slices": len(a.slices),
        },
        "structure": [
            {"section": name, "bars": f"{s}-{e}"} for name, s, e in arrangement.sections
        ],
        "drops": [e for e in arrangement.events if e["type"] == "drop"],
        "wobbles": [e for e in arrangement.events if e["type"] == "wobble"][:8],
        "seed": seed,
    }

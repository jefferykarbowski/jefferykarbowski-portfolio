"""High-level pipeline: analyze -> arrange -> render. Shared by CLI and MCP server."""

from __future__ import annotations

from pathlib import Path

from .engine import analysis, arrange, render


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

    a = analysis.analyze(str(src))
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

"""MCP server exposing the dubstep-mangler engine.

Run with:  dubstep-mangler-mcp   (or `python -m dubstep_mangler.server`)
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from . import mangle as pipeline
from .engine import render, synthesis

mcp = FastMCP(
    "dubstep-mangler",
    instructions=(
        "Tools for slicing a song apart and algorithmically reassembling it as a "
        "dubstep track (golden-ratio structure, Euclidean rhythms, Markov timbre "
        "walks, tempo-locked wobble bass). Start with analyze_song to inspect the "
        "source, then mangle_song to render. All paths are local file paths."
    ),
)


@mcp.tool()
def analyze_song(path: str) -> dict:
    """Analyze an audio file: tempo, musical key, and how it slices apart
    (onset/beat boundaries plus per-slice timbre features)."""
    return pipeline.analyze_summary(path)


@mcp.tool()
def mangle_song(
    input_path: str,
    output_path: str = "",
    tempo: float = 140.0,
    total_bars: int = 48,
    seed: int = 0,
    intensity: float = 1.0,
    wobble_shape: str = "sine",
) -> dict:
    """Slice a song apart and reassemble it as a dubstep track.

    tempo: output BPM (dubstep convention is 140, half-time feel).
    total_bars: length of the arrangement (48 bars at 140 BPM is ~82s).
    seed: same seed = identical track; change it to reroll every algorithmic choice.
    intensity: 0.5 restrained .. 2.0 unhinged (pattern density, wobble speed, drive).
    wobble_shape: LFO shape for the bass, "sine" or "saw".
    Returns the output path plus the generated structure, drop locations, and
    the first few wobble-bass events (frequency and LFO rate per bar).
    """
    return pipeline.mangle(
        input_path,
        output_path or None,
        tempo=tempo,
        total_bars=total_bars,
        seed=seed,
        intensity=intensity,
        wobble_shape=wobble_shape,
    )


@mcp.tool()
def preview_wobble(
    output_path: str,
    root_hz: float = 55.0,
    rate_hz: float = 3.5,
    seconds: float = 4.0,
    shape: str = "sine",
) -> dict:
    """Render an isolated wobble-bass note to a WAV file — useful for auditioning
    LFO rates and shapes before committing to a full mangle."""
    y = synthesis.wobble_bass(root_hz, seconds, rate_hz=rate_hz, lfo_shape=shape)
    render.write(output_path, y, synthesis.SR)
    return {"output_path": output_path, "root_hz": root_hz, "rate_hz": rate_hz}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

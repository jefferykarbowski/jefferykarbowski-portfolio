"""Command-line interface: dubstep-mangler <input> [-o out.wav] [--seed N] ..."""

from __future__ import annotations

import argparse
import json

from . import mangle as pipeline


def main() -> None:
    p = argparse.ArgumentParser(
        prog="dubstep-mangler",
        description="Slice a song apart and reassemble it as a dubstep track.",
    )
    p.add_argument("input", help="source audio file (wav/flac/ogg; mp3 needs ffmpeg)")
    p.add_argument("-o", "--output", help="output WAV path (default: <input>_dubstep_seedN.wav)")
    p.add_argument("--tempo", type=float, default=140.0, help="output BPM (default 140)")
    p.add_argument("--bars", type=int, default=48, help="arrangement length in bars (default 48)")
    p.add_argument("--seed", type=int, default=0, help="reroll all algorithmic choices")
    p.add_argument("--intensity", type=float, default=1.0, help="0.5 restrained .. 2.0 unhinged")
    p.add_argument("--wobble-shape", choices=["sine", "saw"], default="sine")
    p.add_argument("--analyze-only", action="store_true", help="print analysis, render nothing")
    args = p.parse_args()

    if args.analyze_only:
        print(json.dumps(pipeline.analyze_summary(args.input), indent=2))
        return

    result = pipeline.mangle(
        args.input,
        args.output,
        tempo=args.tempo,
        total_bars=args.bars,
        seed=args.seed,
        intensity=args.intensity,
        wobble_shape=args.wobble_shape,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

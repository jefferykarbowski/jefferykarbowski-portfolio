# dubstep-mangler 🎛️

Slice any song apart and algorithmically reassemble it as a dubstep track —
exposed as an **MCP server** so an AI assistant can drive the whole process,
plus a CLI for direct use.

No DAW, no samples, no ML models: pure signal processing (`librosa` analysis,
`numpy`/`scipy` synthesis) driven by generative algorithms.

## The algorithms

| Layer | Algorithm |
|---|---|
| Song structure | Sections cut at **golden-ratio** points (powers of 1/φ): intro → build → drop → breakdown → drop2 → outro |
| Drum & stab placement | **Euclidean rhythms** (Bjorklund) with per-bar rotation |
| Slice sequencing | **Markov random walk** through timbre space — transition probability decays with MFCC/spectral distance, so similar-sounding slices chain together with occasional leaps |
| Buildup density | Hit counts climb the **Fibonacci ladder** for non-linear acceleration |
| Wobble bass | Detuned saws + sine sub through a time-varying lowpass; LFO rates **locked to tempo multiples**, occasionally split mid-bar by Fibonacci ratios |
| Musicality | Source key detected via Krumhansl–Schmuckler chroma profiles; bass follows an i–i–VI–VII (minor) progression on the detected root |

Everything is seeded — same seed, same track; change the seed to reroll every
algorithmic decision.

## How it works

1. **Analyze** — load the song, detect tempo and key, cut it into 100–2000ms
   slices on onset/beat boundaries, fingerprint each slice (MFCC timbre vector,
   spectral centroid, flatness, RMS, onset strength).
2. **Arrange** — sort slices into pools (percussive *stabs*, tonal *beds*),
   then place them into a 140 BPM half-time skeleton alongside synthesized
   kick/snare/hats, wobble bass, risers, sub-drops and impacts.
3. **Render** — sidechain-pump the bass/music against every kick, saturate the
   bus, normalize, write a WAV.

## Install

```bash
cd dubstep-mangler
python3 -m venv .venv && .venv/bin/pip install -e .
```

WAV/FLAC/OGG input works out of the box; MP3 needs `ffmpeg` on PATH.

## CLI

```bash
dubstep-mangler song.wav                    # → song_dubstep_seed0.wav
dubstep-mangler song.wav --seed 7 --intensity 1.6 --wobble-shape saw
dubstep-mangler song.wav --analyze-only     # inspect tempo/key/slices as JSON
```

## MCP server

Register with any MCP client (Claude Code, Claude Desktop, Cursor, ...):

```json
{
  "mcpServers": {
    "dubstep-mangler": {
      "command": "/path/to/dubstep-mangler/.venv/bin/dubstep-mangler-mcp"
    }
  }
}
```

Then ask your assistant things like *"analyze ~/Music/song.wav, then mangle it
at intensity 1.5 with saw wobbles and reroll seeds until the drop lands hard."*

Tools:

- `analyze_song(path)` — tempo, key, slice census and timbre stats
- `mangle_song(input_path, output_path?, tempo?, total_bars?, seed?, intensity?, wobble_shape?)` — render the full track; returns structure, drop timestamps, and per-bar wobble events
- `preview_wobble(output_path, root_hz?, rate_hz?, seconds?, shape?)` — audition an isolated wobble note

## ComfyUI nodes

`comfyui_nodes/` ships a ComfyUI custom-node pack: **Dubstep Mangle** (AUDIO→AUDIO),
**Binaural Beats**, **Isochronic Tones**, and the experimental **Vibration Field**
(golden-ratio partial spirals, Fibonacci gates, Schumann-resonance locks,
breath-paced envelopes), plus an `example_workflow.json` chaining them all.
See [comfyui_nodes/README.md](comfyui_nodes/README.md).

## Try it without a song

```bash
.venv/bin/python scripts/make_test_song.py demo.wav   # synthesizes a test song
.venv/bin/dubstep-mangler demo.wav --bars 32
```

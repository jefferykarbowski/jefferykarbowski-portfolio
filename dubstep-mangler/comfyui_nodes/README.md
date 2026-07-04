# dubstep-mangler ComfyUI nodes

Four audio nodes for ComfyUI that chain into a "mangle a song, then tune the
listener" workflow:

| Node | What it does |
|---|---|
| **Dubstep Mangle 🎛️** | AUDIO → AUDIO. Slices the incoming song apart and reassembles it as a dubstep track (golden-ratio structure, Euclidean rhythms, Markov timbre walks, tempo-locked wobble bass). Second output is the generated structure as JSON. |
| **Binaural Beats 🎧** | Layers a stereo binaural carrier under the audio (or generates standalone). Left ear = carrier, right ear = carrier + beat rate; the rate can sweep, e.g. alpha 10 Hz → theta 6 Hz for a guided descent. Headphones required. |
| **Isochronic Tones 🔊** | Speaker-friendly entrainment: a carrier pulsed on/off at a brainwave-band rate with raised-cosine edges. `gate_music` mode pulses the *music itself* instead of adding a tone. |
| **Vibration Field ✨** | The experimental one. Generators built on mathematical structure rather than tradition — see below. |

## Vibration Field modes

- **phi_spiral** — inharmonic drone with partials spaced at golden-ratio
  (φ^k) intervals and 1/φ^k amplitudes, each breathing at a φ-derived
  sub-hertz rate. Because the partials never align harmonically, the texture
  rotates forever without repeating — a spectrum conventional
  (harmonic-series) instruments cannot produce.
- **fibonacci_gate** — gates audio open only on Fibonacci-numbered steps of a
  34-step cycle: a rhythm of accelerating gaps that resolves once per cycle.
- **schumann_lock** — amplitude-modulates at the Schumann resonance (7.83 Hz)
  plus its golden-ratio partner (≈12.67 Hz); the two rates beat against each
  other at ~4.8 Hz, a slow modulation-of-modulation.
- **breath_pacer** — swells the audio at a coherent-breathing pace
  (5.5 breaths/min, 40% inhale / 60% exhale).
- **full_alchemy** — all of the above stacked: φ-spiral drone → Fibonacci
  gate → Schumann lock → breath pacer, layered under the music (which also
  gets a gentle Schumann/breath treatment).

**Honesty note:** the acoustics are implemented faithfully (the binaural beat
rate is verifiably exact), but scientific evidence that entrainment audio
changes mental states is mixed, and solfeggio/"vibration" framings are
tradition, not physics. Treat these as experimental sound design and
meditation aesthetics — not medicine.

## Install

```bash
# 1. clone the repo anywhere
git clone https://github.com/jefferykarbowski/jefferykarbowski-portfolio
# 2. link the node pack into ComfyUI
ln -s /path/to/jefferykarbowski-portfolio/dubstep-mangler/comfyui_nodes \
      /path/to/ComfyUI/custom_nodes/dubstep-mangler-nodes
# 3. install the engine into ComfyUI's Python environment
/path/to/ComfyUI/python -m pip install -e /path/to/jefferykarbowski-portfolio/dubstep-mangler
```

(If you only symlink and skip step 3, the nodes fall back to importing the
engine from the repo layout, but you still need `pip install -r
requirements.txt` for librosa/scipy/soundfile.)

## Example workflow

Load `example_workflow.json` in ComfyUI:

```
LoadAudio → Dubstep Mangle → Binaural Beats (α→θ sweep)
          → Isochronic Tones (528 Hz @ 10 Hz) → Vibration Field (full_alchemy) → SaveAudio
```

Mangle seed/intensity, beat sweeps, and every Vibration Field mode are
re-rollable and deterministic per seed.

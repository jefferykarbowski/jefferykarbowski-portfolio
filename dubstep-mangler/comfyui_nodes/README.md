# dubstep-mangler ComfyUI nodes

Four audio nodes for ComfyUI that chain into a "mangle a song, then tune the
listener" workflow:

| Node | What it does |
|---|---|
| **Hypnagogic Weave 🌙** | AUDIO → AUDIO. Reassembles the song into slow, evolving, genreless music aimed at the sleep-onset (hypnagogic) state: no drops, an ever-descending arousal curve, consonant modal melody in the song's own key, long reverb tails, a decelerating heartbeat pulse, and a near-inaudible eternal Shepard-tone glide underneath. |
| **Dubstep Mangle 🎛️** | AUDIO → AUDIO. Slices the incoming song apart and reassembles it as a dubstep track (golden-ratio structure, Euclidean rhythms, Markov timbre walks, tempo-locked wobble bass). Second output is the generated structure as JSON. |
| **Binaural Beats (subliminal) 🎧** | Layers a stereo binaural carrier **below conscious detection** (set in dB under the music) so the brain can't fixate on it and habituate. Beat glides along the iso-principle descent (alpha 10.5 Hz → theta 5 Hz) and wanders slightly (anti-habituation drift). Headphones required. |
| **Isochronic Tones 🔊** | Speaker-friendly entrainment: a carrier pulsed on/off at a brainwave-band rate with raised-cosine edges. `gate_music` mode pulses the *music itself* instead of adding a tone. |
| **Vibration Field ✨** | Experimental generators built on mathematical structure rather than tradition — see below. |

## Why the beats are subliminal (and drifting)

Earlier the binaural layer was mixed too loud. Two research-backed reasons it's now quiet and moving:

- **Habituation.** The auditory cortex builds a predictive model of any steady, salient stimulus and then stops tracking it — the frequency-following response fades. A beat that sits *below conscious detection* is felt, not analyzed, so it never becomes a foreground pattern to tune out. The `level_db_below_music` control sets it relative to the program (≈24 dB = felt, 32 = subliminal, 45 = barely there). The literature also warns against *masking* beats with noise — subliminal is about level, not burying them.
- **Anti-habituation drift.** Even subliminally, a perfectly fixed rate is more model-able than a wandering one, so `drift_hz` adds a slow band-limited random walk to the beat frequency. It keeps gently pulling the FFR without ever settling into a predictable pulse.
- **Iso-principle descent.** Entrainment works best starting near the listener's current state, so the beat begins in **alpha (~10.5 Hz, relaxed-alert)** and eases down into **theta (~5 Hz, hypnagogia)** rather than jumping straight to the target.

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

## Example workflows

**`hypnagogic_workflow.json`** — the sleep-onset chain (recommended starting point):

```
LoadAudio → Hypnagogic Weave → Binaural Beats (subliminal, α 10.5 Hz → θ 5 Hz) → SaveAudio
```

**`example_workflow.json`** — the dubstep chain:

```
LoadAudio → Dubstep Mangle → Binaural Beats → Isochronic Tones → Vibration Field (full_alchemy) → SaveAudio
```

Every seed, sweep, scale and mode is re-rollable and deterministic per seed.

> **Note:** Hypnagogic Weave is a heavy offline render (algorithmic reverb +
> per-pad time-stretching) — expect roughly 2× the output duration in render
> time on CPU. It's meant for baking a track, not live preview.

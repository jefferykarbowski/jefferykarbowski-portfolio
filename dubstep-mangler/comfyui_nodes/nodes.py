"""ComfyUI custom nodes: dubstep mangling + entrainment layers.

ComfyUI AUDIO convention: {"waveform": torch.FloatTensor [batch, channels,
samples], "sample_rate": int}. All DSP happens in numpy; torch only at the
boundary.
"""

from __future__ import annotations

import json

import numpy as np

from . import entrainment as ent

ENGINE_SR = 44100


def _to_np(audio: dict) -> tuple[np.ndarray, int]:
    """AUDIO dict -> ((channels, samples) float32 array, sample_rate)."""
    wf = audio["waveform"]
    return wf[0].detach().cpu().numpy().astype(np.float32), int(audio["sample_rate"])


def _to_audio(x: np.ndarray, sr: int) -> dict:
    import torch

    if x.ndim == 1:
        x = x[None, :]
    return {
        "waveform": torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0),
        "sample_rate": sr,
    }


def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0) if x.ndim == 2 else x


def _stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.stack([x, x])
    return x if x.shape[0] == 2 else np.vstack([x, x])[:2]


def _match_len(x: np.ndarray, n: int) -> np.ndarray:
    if x.shape[-1] >= n:
        return x[..., :n]
    pad = [(0, 0)] * (x.ndim - 1) + [(0, n - x.shape[-1])]
    return np.pad(x, pad)


def _mix_under(music: np.ndarray, layer: np.ndarray, layer_gain: float) -> np.ndarray:
    """Blend an entrainment layer under existing audio, soft-limited."""
    music, layer = _stereo(music), _stereo(layer)
    layer = _match_len(layer, music.shape[-1])
    out = music + layer * layer_gain
    peak = np.abs(out).max() or 1.0
    return (np.tanh(out * 0.9) / np.tanh(0.9 * max(peak, 1.0)) * min(peak, 1.0)).astype(
        np.float32
    )


def _carrier_from(preset: str, custom_hz: float) -> float:
    return ent.SOLFEGGIO_PRESETS.get(preset, custom_hz)


def _beat_from(preset: str, custom_hz: float) -> float:
    return ent.BRAINWAVE_PRESETS.get(preset, custom_hz)


class DubstepMangle:
    """Slice the incoming song apart and reassemble it as a dubstep track
    (golden-ratio structure, Euclidean rhythms, Markov timbre walks,
    tempo-locked wobble bass)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "tempo": ("FLOAT", {"default": 140.0, "min": 70.0, "max": 200.0, "step": 1.0}),
                "total_bars": ("INT", {"default": 48, "min": 16, "max": 128, "step": 2}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 2.5, "step": 0.05}),
                "wobble_shape": (["sine", "saw"],),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "structure_json")
    FUNCTION = "mangle"
    CATEGORY = "audio/dubstep-mangler"

    def mangle(self, audio, tempo, total_bars, seed, intensity, wobble_shape):
        import librosa

        from dubstep_mangler.mangle import mangle_audio

        x, sr = _to_np(audio)
        y = _mono(x)
        if sr != ENGINE_SR:
            y = librosa.resample(y.astype(np.float64), orig_sr=sr, target_sr=ENGINE_SR)
        mix, out_sr, meta = mangle_audio(
            y,
            ENGINE_SR,
            tempo=tempo,
            total_bars=total_bars,
            seed=seed,
            intensity=intensity,
            wobble_shape=wobble_shape,
        )
        return (_to_audio(mix, out_sr), json.dumps(meta, indent=2))


class BinauralBeats:
    """Subliminal binaural beats (headphones required), layered under incoming
    audio at a level *below* conscious detection so the brain can't lock onto
    them as a pattern and habituate. The beat glides along the iso-principle
    descent (e.g. alpha 10.5 Hz -> theta 5 Hz) and wanders slightly
    (anti-habituation drift). Left ear = carrier, right ear = carrier + beat."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "carrier_preset": (["custom"] + list(ent.SOLFEGGIO_PRESETS), {"default": "custom"}),
                "carrier_hz": ("FLOAT", {"default": 200.0, "min": 40.0, "max": 1500.0,
                                         "tooltip": "lower carriers (150-250 Hz) give a clearer beat"}),
                "beat_start_preset": (["custom"] + list(ent.BRAINWAVE_PRESETS), {"default": "alpha (relaxation, 10 Hz)"}),
                "beat_start_hz": ("FLOAT", {"default": 10.5, "min": 0.5, "max": 45.0, "step": 0.1}),
                "beat_end_preset": (["custom"] + list(ent.BRAINWAVE_PRESETS), {"default": "theta (meditation, 6 Hz)"}),
                "beat_end_hz": ("FLOAT", {"default": 5.0, "min": 0.5, "max": 45.0, "step": 0.1}),
                "level_db_below_music": ("FLOAT", {"default": 32.0, "min": 6.0, "max": 60.0, "step": 1.0,
                                                   "tooltip": "how far under the music RMS: 24=felt, 32=subliminal, 45=barely there"}),
                "drift_hz": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0, "step": 0.1,
                                       "tooltip": "anti-habituation wander; 0 = perfectly steady (not recommended)"}),
                "curve": (["ease", "linear", "slow_settle"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "standalone_gain": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                                              "tooltip": "output level when NO audio is connected"}),
                "duration_s": ("FLOAT", {"default": 90.0, "min": 1.0, "max": 3600.0}),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "audio/dubstep-mangler"

    def generate(self, carrier_preset, carrier_hz, beat_start_preset, beat_start_hz,
                 beat_end_preset, beat_end_hz, level_db_below_music, drift_hz, curve,
                 seed, standalone_gain, duration_s, audio=None):
        carrier = _carrier_from(carrier_preset, carrier_hz)
        start = _beat_from(beat_start_preset, beat_start_hz)
        end = _beat_from(beat_end_preset, beat_end_hz)
        if audio is not None:
            x, sr = _to_np(audio)
            tone = ent.binaural_beats(x.shape[-1], sr, carrier, start, end, drift_hz, curve, seed)
            return (_to_audio(ent.mix_subliminal(_stereo(x), tone, level_db_below_music), sr),)
        sr = ENGINE_SR
        tone = ent.binaural_beats(int(duration_s * sr), sr, carrier, start, end, drift_hz, curve, seed)
        return (_to_audio(tone * standalone_gain, sr),)


class IsochronicTones:
    """Isochronic pulses: a carrier switched on/off at a brainwave-band rate
    (works on speakers). `gate_music` mode pulses the incoming audio itself
    instead of adding a tone."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["tone", "gate_music"],),
                "carrier_preset": (["custom"] + list(ent.SOLFEGGIO_PRESETS), {"default": "MI 528 Hz (transformation)"}),
                "carrier_hz": ("FLOAT", {"default": 528.0, "min": 40.0, "max": 1500.0}),
                "rate_preset": (["custom"] + list(ent.BRAINWAVE_PRESETS), {"default": "alpha (relaxation, 10 Hz)"}),
                "rate_start_hz": ("FLOAT", {"default": 10.0, "min": 0.5, "max": 45.0, "step": 0.1}),
                "rate_end_hz": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 45.0, "step": 0.1}),
                "duty": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05}),
                "gain": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "duration_s": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 3600.0}),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "audio/dubstep-mangler"

    def generate(self, mode, carrier_preset, carrier_hz, rate_preset, rate_start_hz,
                 rate_end_hz, duty, gain, duration_s, audio=None):
        carrier = _carrier_from(carrier_preset, carrier_hz)
        start = _beat_from(rate_preset, rate_start_hz)
        end = rate_end_hz if rate_end_hz > 0 else start
        if mode == "gate_music" and audio is not None:
            x, sr = _to_np(audio)
            return (_to_audio(ent.pulse_gate(x, sr, start, duty), sr),)
        if audio is not None:
            x, sr = _to_np(audio)
            tone = ent.isochronic_tones(x.shape[-1], sr, carrier, start, end, duty)
            return (_to_audio(_mix_under(x, tone, gain), sr),)
        sr = ENGINE_SR
        tone = ent.isochronic_tones(int(duration_s * sr), sr, carrier, start, end, duty)
        return (_to_audio(tone * gain, sr),)


class VibrationField:
    """Experimental generators built on mathematical structure rather than
    tradition: golden-ratio (phi) inharmonic partial spirals, Fibonacci gate
    rhythms, Schumann-resonance modulation locks, and coherent-breathing
    envelopes. `full_alchemy` stacks all of them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["phi_spiral", "fibonacci_gate", "schumann_lock",
                          "breath_pacer", "full_alchemy"],),
                "base_preset": (["custom"] + list(ent.SOLFEGGIO_PRESETS), {"default": "MI 528 Hz (transformation)"}),
                "base_hz": ("FLOAT", {"default": 111.0, "min": 30.0, "max": 1000.0}),
                "depth": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "gain": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "duration_s": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 3600.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "audio/dubstep-mangler"

    def generate(self, mode, base_preset, base_hz, depth, gain, duration_s, seed, audio=None):
        base = _carrier_from(base_preset, base_hz)
        if audio is not None:
            x, sr = _to_np(audio)
            n = x.shape[-1]
        else:
            x, sr = None, ENGINE_SR
            n = int(duration_s * sr)

        drone = ent.phi_spiral_stack(n, sr, base, shimmer=depth * 2, seed=seed)

        if mode == "phi_spiral":
            layer = drone
        elif mode == "fibonacci_gate":
            layer = ent.fibonacci_gate(x if x is not None else drone, sr)
            if x is not None:  # gated the music itself — that IS the output
                return (_to_audio(layer, sr),)
        elif mode == "schumann_lock":
            if x is not None:
                return (_to_audio(ent.schumann_lock(x, sr, depth), sr),)
            layer = ent.schumann_lock(drone, sr, depth)
        elif mode == "breath_pacer":
            if x is not None:
                return (_to_audio(ent.breath_pacer(x, sr, depth=depth), sr),)
            layer = ent.breath_pacer(drone, sr, depth=depth)
        else:  # full_alchemy: phi drone -> fib gate -> schumann AM -> breath swell
            layer = ent.breath_pacer(
                ent.schumann_lock(ent.fibonacci_gate(drone, sr), sr, depth), sr, depth=depth
            )
            if x is not None:
                music = ent.breath_pacer(ent.schumann_lock(x, sr, depth * 0.5), sr, depth=depth * 0.5)
                return (_to_audio(_mix_under(music, layer, gain), sr),)

        if x is not None:
            return (_to_audio(_mix_under(x, layer, gain), sr),)
        return (_to_audio(layer * gain, sr),)


class HypnagogicWeave:
    """Reassemble the incoming song into slow, evolving, genreless music built
    to guide the listener toward the hypnagogic (sleep-onset) state: no drops,
    an ever-descending arousal curve, consonant modal melody in the song's own
    key, long reverb tails, a decelerating heartbeat pulse, and a near-inaudible
    eternal Shepard-tone glide underneath. Pair its output with a subliminal
    Binaural Beats node for the entrainment layer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "duration_s": ("FLOAT", {"default": 90.0, "min": 20.0, "max": 1800.0, "step": 5.0}),
                "scale": (["pentatonic_minor", "pentatonic_major", "dorian", "lydian"],),
                "pulse_start_bpm": ("FLOAT", {"default": 66.0, "min": 40.0, "max": 90.0, "step": 1.0}),
                "pulse_end_bpm": ("FLOAT", {"default": 50.0, "min": 30.0, "max": 90.0, "step": 1.0,
                                            "tooltip": "heartbeat pulse decelerates to this by the end"}),
                "depth": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.05,
                                    "tooltip": "reverb/delay/shimmer lushness"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "structure_json")
    FUNCTION = "weave"
    CATEGORY = "audio/dubstep-mangler"

    def weave(self, audio, duration_s, scale, pulse_start_bpm, pulse_end_bpm, depth, seed):
        import librosa

        from dubstep_mangler.mangle import weave_audio

        x, sr = _to_np(audio)
        y = _mono(x)
        if sr != ENGINE_SR:
            y = librosa.resample(y.astype(np.float64), orig_sr=sr, target_sr=ENGINE_SR)
        out, out_sr, meta = weave_audio(
            y, ENGINE_SR, duration_s=duration_s, seed=seed, scale=scale,
            pulse_start_bpm=pulse_start_bpm, pulse_end_bpm=pulse_end_bpm, depth=depth,
        )
        return (_to_audio(out, out_sr), json.dumps(meta, indent=2))


NODE_CLASS_MAPPINGS = {
    "DubstepMangle": DubstepMangle,
    "HypnagogicWeave": HypnagogicWeave,
    "BinauralBeats": BinauralBeats,
    "IsochronicTones": IsochronicTones,
    "VibrationField": VibrationField,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DubstepMangle": "Dubstep Mangle 🎛️",
    "HypnagogicWeave": "Hypnagogic Weave 🌙",
    "BinauralBeats": "Binaural Beats (subliminal) 🎧",
    "IsochronicTones": "Isochronic Tones 🔊",
    "VibrationField": "Vibration Field ✨",
}

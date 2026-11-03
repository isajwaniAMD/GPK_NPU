"""In-car voice-assistant test clip WITH OVERLAPPING SPEECH: the same 4 speakers (US English, Piper TTS) ask the car assistant the
same questions as incar_commands_us_en.wav, but several of them start talking before the previous speaker has finished.
Optionally a music file is mixed in as a very quiet background bed: it fades in smoothly, plays alone for a few seconds, keeps
running under the whole conversation and fades out after the last word.

Run with the Piper venv (from this folder):
  # no background
  ~/isdev/tts_tools/venv/bin/python make_incar_overlap_wav.py ../audio_files/incar_overlap_us_en.wav ../eval/data/incar_overlap_us_en.txt
  # with quiet background music
  ~/isdev/tts_tools/venv/bin/python make_incar_overlap_wav.py ../audio_files/incar_overlap_music_us_en.wav \
      ../eval/data/incar_overlap_music_us_en.txt --music ../audio_files/rivian_amd_anthem.wav
Output: 16 kHz mono 16-bit WAV + ground truth lines "[start - end] Speaker N: text" with the TRUE start/end of every line
(overlapping lines overlap in time; with --music the times include the lead-in), and a summary of how much speech is overlapped.
"""
import argparse
import subprocess
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice

VOICES = Path.home() / "isdev" / "tts_tools" / "voices"
# Speaker 1 driver (lessac, F), 2 front passenger (ryan, M), 3 rear passenger (amy, F), 4 rear passenger (joe, M)
SPEAKERS = {"Speaker 1": "lessac", "Speaker 2": "ryan", "Speaker 3": "amy", "Speaker 4": "joe"}
# (speaker, text, gap): a line starts `gap` seconds after the previous line ENDS; a negative gap means it starts while the
# previous speaker is still talking.
SCRIPT = [
    ("Speaker 1", "Hey car, what is the quickest route to the nearest charging station?", 0.0),
    ("Speaker 2", "Can you also tell me how much battery range we have left?", -1.2),
    ("Speaker 3", "Please turn the temperature in the back seat down a little, it is too warm.", 0.6),
    ("Speaker 4", "What is the capital of Australia? We are arguing about it.", -2.0),
    ("Speaker 1", "Remind me to buy groceries when we get home this evening.", -1.0),
    ("Speaker 2", "Explain in one sentence why electric cars have regenerative braking.", 0.7),
    ("Speaker 3", "Can you suggest a relaxing playlist for a long drive?", -1.5),
    ("Speaker 4", "How long does it usually take to fully charge a car like this?", -2.5),
    ("Speaker 1", "Thanks. And what is the speed limit on a typical highway in California?", 0.6),
    ("Speaker 2", "Tell me a short joke to keep the kids entertained.", -1.3),
]
SPEECH_RMS = 0.1  # every line is normalised to this RMS so overlaps are balanced

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("out_wav", type=Path)
ap.add_argument("out_txt", type=Path)
ap.add_argument("--music", type=Path, help="background music file (anything ffmpeg reads); looped if shorter than the clip")
ap.add_argument("--music-db", type=float, default=-22.0, help="music loudness relative to the voices in dB (default -22, very quiet)")
ap.add_argument("--lead-in", type=float, help="seconds of music alone before the first word (default 4 with --music, else 0)")
ap.add_argument("--fade-in", type=float, default=3.0, help="seconds for the music to fade in smoothly (default 3)")
ap.add_argument("--fade-out", type=float, default=3.0, help="seconds for the music to fade out at the end (default 3)")
ap.add_argument("--tail", type=float, help="seconds after the last word (default 3 with --music, else 0.5)")
args = ap.parse_args()
lead = args.lead_in if args.lead_in is not None else (4.0 if args.music else 0.0)
tail = args.tail if args.tail is not None else (3.0 if args.music else 0.5)

voices = {s: PiperVoice.load(VOICES / f"en_US-{v}-medium.onnx") for s, v in SPEAKERS.items()}
rate = next(iter(voices.values())).config.sample_rate


def synth(spk, text):
    a = np.concatenate([c.audio_int16_array for c in voices[spk].synthesize(text)]).astype(np.float32) / 32768.0
    return a * (SPEECH_RMS / max(np.sqrt(np.mean(a ** 2)), 1e-6))


def music_bed(path, n_samples):
    """Music as mono float at the speech rate, scaled to --music-db below the voices, with smooth (raised-cosine) fades."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    m = np.frombuffer(raw, dtype=np.float32)
    m = np.tile(m, -(-n_samples // len(m)))[:n_samples].copy()
    m *= SPEECH_RMS * 10 ** (args.music_db / 20) / max(np.sqrt(np.mean(m ** 2)), 1e-6)
    n_in, n_out = min(int(args.fade_in * rate), n_samples), min(int(args.fade_out * rate), n_samples)
    m[:n_in] *= 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, n_in))
    m[n_samples - n_out:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, n_out))
    return m


lines, prev_end = [], 0.0
for spk, text, gap in SCRIPT:
    audio = synth(spk, text)
    start = max(0.0, prev_end + gap)
    end = start + len(audio) / rate
    lines.append((start, end, spk, text, audio))
    prev_end = end
total = int((lead + prev_end + tail) * rate)
mix = np.zeros(total, dtype=np.float32)
for start, _, _, _, audio in lines:
    i = int((lead + start) * rate)
    mix[i:i + len(audio)] += audio
if args.music:
    mix += music_bed(args.music, total)
peak = np.abs(mix).max()
if peak > 0.9:
    mix *= 0.9 / peak
pcm = (mix * 32767).astype("<i2")

# overlap statistics: seconds with 2+ (and 3+) simultaneous speakers
events = sorted({0.0, *(l[0] for l in lines), *(l[1] for l in lines)})
active = lambda a, b: sum(l[0] < b and l[1] > a for l in lines)
speech_s = sum(b - a for a, b in zip(events, events[1:]) if active(a, b) >= 1)
overlap_s = sum(b - a for a, b in zip(events, events[1:]) if active(a, b) >= 2)
three_s = sum(b - a for a, b in zip(events, events[1:]) if active(a, b) >= 3)
truth = [f"[{lead + s:6.1f} - {lead + e:6.1f}] {spk}: {text}" for s, e, spk, text, _ in lines]
tmp = args.out_wav.with_suffix(".tmp.wav")
with wave.open(str(tmp), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-ar", "16000", "-ac", "1", str(args.out_wav)], check=True)
tmp.unlink()
args.out_txt.write_text("\n".join(truth) + "\n")
print("\n".join(truth))
print(f"\n{args.out_wav}: {total / rate:.1f}s, speech {speech_s:.1f}s, overlapped (2+ speakers) {overlap_s:.1f}s = "
      f"{overlap_s / speech_s * 100:.0f}% of speech, 3+ speakers {three_s:.1f}s"
      + (f"; music {args.music_db:+.0f} dB under the voices, {lead:.1f}s lead-in" if args.music else ""))

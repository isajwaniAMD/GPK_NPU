"""In-car voice-assistant test clip: 4 speakers (US English, Piper TTS, clean, no overlap) ask the car assistant questions.

Run with the Piper venv:  ~/isdev/tts_tools/venv/bin/python make_incar_wav.py incar_commands_us_en.wav incar_commands_us_en.txt
Output: 16 kHz mono 16-bit WAV + ground truth lines "[start - end] Speaker N: text".
"""
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from piper import PiperVoice

VOICES = Path.home() / "isdev" / "tts_tools" / "voices"
# Speaker 1 driver (lessac, F), 2 front passenger (ryan, M), 3 rear passenger (amy, F), 4 rear passenger (joe, M)
SPEAKERS = {"Speaker 1": "lessac", "Speaker 2": "ryan", "Speaker 3": "amy", "Speaker 4": "joe"}
GAP = 0.8  # seconds of silence between turns
SCRIPT = [
    ("Speaker 1", "Hey car, what is the quickest route to the nearest charging station?"),
    ("Speaker 2", "Can you also tell me how much battery range we have left?"),
    ("Speaker 3", "Please turn the temperature in the back seat down a little, it is too warm."),
    ("Speaker 4", "What is the capital of Australia? We are arguing about it."),
    ("Speaker 1", "Remind me to buy groceries when we get home this evening."),
    ("Speaker 2", "Explain in one sentence why electric cars have regenerative braking."),
    ("Speaker 3", "Can you suggest a relaxing playlist for a long drive?"),
    ("Speaker 4", "How long does it usually take to fully charge a car like this?"),
    ("Speaker 1", "Thanks. And what is the speed limit on a typical highway in California?"),
    ("Speaker 2", "Tell me a short joke to keep the kids entertained."),
]

out_wav, out_txt = Path(sys.argv[1]), Path(sys.argv[2])
voices = {s: PiperVoice.load(VOICES / f"en_US-{v}-medium.onnx") for s, v in SPEAKERS.items()}
rate = next(iter(voices.values())).config.sample_rate
parts, t, truth = [], 0.0, []
for spk, text in SCRIPT:
    audio = np.concatenate([c.audio_int16_array for c in voices[spk].synthesize(text)])
    truth.append(f"[{t:6.1f} - {t + len(audio) / rate:6.1f}] {spk}: {text}")
    t += len(audio) / rate + GAP
    parts += [audio, np.zeros(int(GAP * rate), dtype=np.int16)]
tmp = out_wav.with_suffix(".tmp.wav")
with wave.open(str(tmp), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(np.concatenate(parts).tobytes())
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-ar", "16000", "-ac", "1", str(out_wav)], check=True)
tmp.unlink()
out_txt.write_text("\n".join(truth) + "\n")
print("\n".join(truth)); print(f"\n{out_wav}: {t:.1f}s")

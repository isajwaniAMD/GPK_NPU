"""Build synthetic multi-speaker conversations with exact ground-truth speaker times.

Usage: python make_test_audio.py [seed ...]   (seed 0 writes synthetic_5spk.wav, others synthetic_5spk_s<seed>.wav)
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
OUT_DIR = Path(__file__).parent / "audio"

SPEAKERS = {
    "A": [
        r"C:\Dev\RyzenAI-SW\demo\ASR\Whisper\eval_dataset\LibriSpeech-samples\wav\1089-134686-0009.wav",
        r"C:\Dev\RyzenAI-SW\demo\ASR\Whisper\audio_files\1089-134686-0000.wav",
    ],
    "B": [r"C:\Dev\RyzenAI-SW\demo\ASR\Whisper\eval_dataset\LibriSpeech-samples\wav\3570-5694-0000.wav"],
    "C": [r"C:\Dev\whisper.cpp\samples\jfk.wav"],
    "D": [r"C:\Dev\RyzenAI-SW\demo\ASR\Whisper\audio_files\61-52s.wav"],
    "E": [r"C:\Dev\WhisperKit\Tests\WhisperKitTests\Resources\es_test_clip.wav"],
}


def load_speaker(paths):
    audio = np.concatenate([sf.read(p, dtype="float32")[0] for p in paths])
    return audio / (np.sqrt(np.mean(audio**2)) + 1e-8) * 0.05


def split_pieces(audio, rng):
    pieces, pos = [], 0
    while pos < len(audio):
        length = int(rng.uniform(3.0, 6.0) * SR)
        piece = audio[pos : pos + length]
        if len(piece) > SR:
            pieces.append(piece)
        pos += length
    return pieces


def build(seed):
    rng = np.random.default_rng(seed)
    queues = {name: split_pieces(load_speaker(paths), rng) for name, paths in SPEAKERS.items()}

    placed, cursor, previous = [], 0.5, None
    while any(queues.values()):
        candidates = [name for name, q in queues.items() if q and name != previous] or [
            name for name, q in queues.items() if q
        ]
        name = candidates[rng.integers(len(candidates))]
        piece = queues[name].pop(0)
        # a quarter of the turns barge in on the previous speaker
        offset = -rng.uniform(0.5, 1.2) if previous and rng.random() < 0.25 else rng.uniform(0.2, 1.0)
        start = max(0.0, cursor + offset)
        placed.append((start, piece, name))
        cursor = start + len(piece) / SR
        previous = name

    total = int((max(s + len(p) / SR for s, p, _ in placed) + 0.5) * SR)
    mix = np.zeros(total, dtype=np.float32)
    truth = []
    for start, piece, name in placed:
        i = int(start * SR)
        mix[i : i + len(piece)] += piece
        truth.append({"start": round(start, 3), "end": round(start + len(piece) / SR, 3), "speaker": name})
    mix /= max(1.0, np.abs(mix).max() / 0.95)

    OUT_DIR.mkdir(exist_ok=True)
    name = "synthetic_5spk" if seed == 0 else f"synthetic_5spk_s{seed}"
    sf.write(OUT_DIR / f"{name}.wav", mix, SR, subtype="PCM_16")
    (OUT_DIR / f"{name}.truth.json").write_text(json.dumps(truth, indent=1))
    print(f"{name}.wav: {total / SR:.1f}s, {len(truth)} turns, speakers {sorted(SPEAKERS)}")


if __name__ == "__main__":
    for seed in [int(s) for s in sys.argv[1:]] or [0]:
        build(seed)

"""Run the official Transformers Nemotron-3-Diarization model and save reference outputs.

Usage: python reference_hf.py audio/synthetic_5spk.wav [more.wav ...]
Writes <audio>.ref.npz with the mel features and per-frame speaker probabilities (10 ms frames).
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

MODEL_DIR = Path(__file__).parent / "models" / "hf"


def main():
    torch.set_num_threads(8)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL_DIR, attn_implementation="eager").eval()

    for path in sys.argv[1:]:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        inputs = processor(audio, sampling_rate=sr)

        with torch.inference_mode():
            start = time.perf_counter()
            logits = model(**inputs).logits
            elapsed = time.perf_counter() - start

        probs = logits.sigmoid()[0].numpy()
        segments = processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
        out = Path(path).with_suffix(".ref.npz")
        np.savez(out, features=inputs.input_features[0].numpy(), probs=probs)

        duration = len(audio) / sr
        print(f"{path}: {duration:.1f}s audio, {elapsed:.2f}s on CPU (RTFx {duration / elapsed:.0f}), "
              f"{len(segments)} segments, speakers {sorted({s['Speaker'] for s in segments})} -> {out.name}")
        for s in segments[:12]:
            print(f"  {s['Start']:7.2f} {s['End']:7.2f} speaker_{s['Speaker']}")


if __name__ == "__main__":
    main()

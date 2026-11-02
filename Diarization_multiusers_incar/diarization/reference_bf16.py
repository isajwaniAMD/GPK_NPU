"""How much does BF16 alone change the PyTorch output? Compares BF16 runs against the FP32 reference.

Usage: python reference_bf16.py audio/synthetic_5spk.wav
"""

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

MODEL_DIR = Path(__file__).parent / "models" / "hf"


def report(name, probs, ref, mask):
    n = min(len(probs), len(ref))
    diff = np.abs(probs[:n] - ref[:n])
    agree = ((probs[:n] > 0.5) == (ref[:n] > 0.5)).all(axis=1)[mask[:n]].mean()
    print(f"  {name:28s} max |diff| {diff.max():.3f}, mean {diff.mean():.4f}, identical decisions {agree * 100:.2f}%")


def main():
    torch.set_num_threads(8)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL_DIR, attn_implementation="eager").eval()

    for path in sys.argv[1:]:
        audio, sr = sf.read(path, dtype="float32")
        inputs = processor(audio, sampling_rate=sr)
        ref = np.load(Path(path).with_suffix(".ref.npz"))["probs"]
        mask = inputs.attention_mask[0].numpy()
        print(path)
        with torch.inference_mode():
            with torch.autocast("cpu", dtype=torch.bfloat16):
                probs = model(**inputs).logits.float().sigmoid()[0].numpy()
            report("autocast bf16 (NeMo-style)", probs, ref, mask)

            bf16 = AutoModelForAudioFrameClassification.from_pretrained(
                MODEL_DIR, attn_implementation="eager", dtype=torch.bfloat16
            ).eval()
            bf16_inputs = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in inputs.items()}
            probs = bf16(**bf16_inputs).logits.float().sigmoid()[0].numpy()
            report("pure bf16 weights+activations", probs, ref, mask)
            np.savez(Path(path).with_suffix(".bf16.npz"), probs=probs)


if __name__ == "__main__":
    main()

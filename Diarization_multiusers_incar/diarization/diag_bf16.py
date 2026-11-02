"""Is the NPU error BF16-sized? Runs the step network in PyTorch BF16 inside the numpy pipeline.

Reports per-step error of the BF16 network on the FP32 pipeline's inputs (compare with diag_step.py on the NPU),
then the end-to-end output of the pipeline driven by the BF16 network (compare with the NPU run).

Usage: python diag_bf16.py audio/synthetic_5spk.wav
"""

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForAudioFrameClassification

import diarizer as dz
from export_onnx import DiarizationStep


def main():
    torch.set_num_threads(8)
    path = sys.argv[1]
    audio = dz.load_audio(path)
    model = AutoModelForAudioFrameClassification.from_pretrained(
        dz.MODELS / "hf", attn_implementation="eager", dtype=torch.bfloat16
    ).eval()
    step = DiarizationStep(model, 704).to(torch.bfloat16).eval()

    def run_bf16(embeds, mask):
        with torch.inference_mode():
            out = step(torch.from_numpy(embeds).to(torch.bfloat16), torch.from_numpy(mask).to(torch.bfloat16))
        return out.float().numpy()

    cpu = dz.Diarizer("cpu")
    recorded = []
    run_cpu = cpu._run_step

    def record(embeds, mask):
        out = run_cpu(embeds, mask)
        recorded.append((embeds.copy(), mask.copy(), out))
        return out

    cpu._run_step = record
    cpu.diarize(audio)
    for i, (embeds, mask, ref) in enumerate(recorded):
        valid = int(mask.sum()) * dz.SUBSAMPLING
        ref, out = ref[0, :valid], run_bf16(embeds, mask)[0, :valid]
        diff = np.abs(ref - out)
        agree = ((ref > 0.5) == (out > 0.5)).all(axis=1).mean()
        print(f"bf16 step {i}: max |diff| {diff.max():.3f}, mean {diff.mean():.4f}, "
              f"identical decisions {agree * 100:.2f}%")

    cpu._run_step = run_bf16
    probs, _, _ = cpu.diarize(audio)
    out = Path("runs") / (Path(path).stem + ".pipeline_bf16.npz")
    np.savez(out, probs=probs)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

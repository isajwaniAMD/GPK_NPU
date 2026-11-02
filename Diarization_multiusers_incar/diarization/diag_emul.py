"""Emulate NPU number formats on CPU to find which one explains the NPU's per-step error.

Modes: fp32, bf16 (round every matmul operand to BF16), bfp16 (block floating point: blocks of 8 values along the
reduction axis share the largest exponent, 8-bit signed mantissas, as the NPU's bfp16ebs8 format).

Usage: python diag_emul.py audio/synthetic_5spk.wav
"""

import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForAudioFrameClassification
from transformers.models.nemotron3_diarization import modeling_nemotron3_diarization as modeling

import diarizer as dz
from export_onnx import DiarizationStep

MODE = "fp32"


def to_bf16(x):
    return x.to(torch.bfloat16).to(torch.float32)


def to_bfp16(x, dim=-1, block=8):
    """Block floating point along `dim`: shared exponent per `block` values, 8-bit signed mantissa."""
    x = x.movedim(dim, -1)
    shape = x.shape
    pad = -shape[-1] % block
    xb = torch.nn.functional.pad(x, (0, pad)).reshape(*shape[:-1], -1, block)
    max_abs = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    scale = torch.exp2(torch.floor(torch.log2(max_abs)) - 6)
    xq = (torch.round(xb / scale).clamp(-127, 127) * scale).reshape(*shape[:-1], -1)[..., : shape[-1]]
    return xq.movedim(-1, dim)


def quant(x, dim=-1):
    if MODE == "bf16":
        return to_bf16(x)
    if MODE in ("bfp16", "bfp16_act"):
        return to_bfp16(x, dim)
    return x


def quant_w(w, dim=-1):
    return to_bf16(w) if MODE == "bfp16_act" else quant(w, dim)


original_linear = torch.nn.functional.linear


def linear(x, weight, bias=None):
    return original_linear(quant(x, -1), quant_w(weight, -1), bias)


def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    scores = torch.matmul(quant(query, -1), quant(key, -1).transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(quant(weights, -1), quant(value, -2))
    return out.transpose(1, 2).contiguous(), weights


def main():
    global MODE
    torch.set_num_threads(8)
    path = sys.argv[1]
    audio = dz.load_audio(path)
    model = AutoModelForAudioFrameClassification.from_pretrained(dz.MODELS / "hf", attn_implementation="eager").eval()
    step = DiarizationStep(model, 704).eval()
    torch.nn.functional.linear = linear
    modeling.eager_attention_forward = attention

    def run(embeds, mask):
        with torch.inference_mode():
            return step(torch.from_numpy(embeds), torch.from_numpy(mask)).numpy()

    cpu = dz.Diarizer("cpu")
    recorded = []
    run_cpu = cpu._run_step

    def record(embeds, mask):
        out = run_cpu(embeds, mask)
        recorded.append((embeds.copy(), mask.copy(), out))
        return out

    cpu._run_step = record
    cpu.diarize(audio)

    for MODE in ("fp32", "bf16", "bfp16_act", "bfp16"):
        errors = []
        for embeds, mask, ref in recorded:
            valid = int(mask.sum()) * dz.SUBSAMPLING
            errors.append(np.abs(ref[0, :valid] - run(embeds, mask)[0, :valid]).mean())
        cpu._run_step = run
        probs, _, _ = cpu.diarize(audio)
        out = Path("runs") / f"{Path(path).stem}.emul_{MODE}.npz"
        np.savez(out, probs=probs)
        print(f"{MODE:10s} per-step mean |diff|: {' '.join(f'{e:.4f}' for e in errors)}  -> {out.name}")


if __name__ == "__main__":
    main()

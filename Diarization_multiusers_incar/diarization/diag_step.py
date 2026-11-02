"""Compare the NPU and CPU step networks on identical inputs, to separate network error from cache drift.

Usage: python diag_step.py audio/synthetic_5spk.wav [--parts 1]
"""

import argparse

import numpy as np

import diarizer as dz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--vai-config", default="vai_ep_config_accurate.json")
    args = parser.parse_args()

    audio = dz.load_audio(args.audio)
    cpu = dz.Diarizer("cpu", parts=args.parts)
    npu = dz.Diarizer("npu", parts=args.parts, vai_config=args.vai_config)

    recorded = []
    run_cpu = cpu._run_step

    def record(embeds, mask):
        out = run_cpu(embeds, mask)
        recorded.append((embeds.copy(), mask.copy(), out))
        return out

    cpu._run_step = record
    cpu.diarize(audio)

    for i, (embeds, mask, ref) in enumerate(recorded):
        out = npu._run_step(embeds, mask)
        valid = int(mask.sum()) * dz.SUBSAMPLING
        ref, out = ref[0, :valid], out[0, :valid]
        diff = np.abs(ref - out)
        agree = ((ref > 0.5) == (out > 0.5)).all(axis=1).mean()
        print(f"step {i}: {valid // 8} valid frames, max |diff| {diff.max():.3f}, mean {diff.mean():.4f}, "
              f"identical decisions {agree * 100:.2f}%")


if __name__ == "__main__":
    main()

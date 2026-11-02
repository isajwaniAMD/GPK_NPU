"""Score the streaming diarizer: DER vs the Transformers streaming reference and vs ground truth.

Usage: python eval_stream.py [--device npu]   (saves runs/<clip>.stream_<device>.npz)
"""

import argparse
import glob
import os

import numpy as np

import diarizer as dz
from eval_der import der, truth_matrix
from stream_diarizer import StreamingDiarizer


def active(probs):
    hyp = probs > 0.5
    return hyp[:, hyp.any(axis=0)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu")
    args = parser.parse_args()

    clips = sorted(glob.glob("audio/synthetic_5spk*.wav")) + ["audio/diarization_example.wav"]
    stream_errors, offline_errors = [], []
    for path in clips:
        name = os.path.basename(path)[:-4]
        stream = StreamingDiarizer(args.device)
        audio = dz.load_audio(path)
        for i in range(0, len(audio), dz.SAMPLE_RATE):
            stream.feed(audio[i : i + dz.SAMPLE_RATE])
        probs = stream.probs()
        np.savez(f"runs/{name}.stream_{args.device}.npz", probs=probs)

        row = f"{name:22s}"
        ref_path = f"audio/{name}.stream_low_latency.ref.npz"
        if os.path.exists(ref_path):
            ref = np.load(ref_path)["probs"][: len(probs)]
            row += f"  vs streaming reference {der(active(ref), active(probs[: len(ref)]))[0] * 100:6.2f}%"
        truth_path = f"audio/{name}.truth.json"
        if os.path.exists(truth_path):
            stream_error = der(truth_matrix(truth_path, len(probs)), active(probs))[0]
            offline = np.load(f"runs/{name}.cpu.npz")["probs"]
            offline_error = der(truth_matrix(truth_path, len(offline)), active(offline))[0]
            stream_errors.append(stream_error)
            offline_errors.append(offline_error)
            row += f"  vs truth: streaming {stream_error * 100:6.2f}%  (offline CPU {offline_error * 100:6.2f}%)"
        print(row, flush=True)
    print(f"{'mean vs truth':22s}  streaming {np.mean(stream_errors) * 100:6.2f}%  "
          f"(offline CPU {np.mean(offline_errors) * 100:6.2f}%)")


if __name__ == "__main__":
    main()

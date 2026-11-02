"""Frame-level diarization error rate (no collar, overlap scored) with the best speaker mapping.

Usage:
  python eval_der.py audio/synthetic_5spk.truth.json run1.npz run2.npz ...   (ground truth)
  python eval_der.py audio/diarization_example.ref.npz run1.npz ...          (FP32 reference as truth)
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

FRAME = 0.01


def truth_matrix(path, num_frames):
    turns = json.loads(Path(path).read_text())
    speakers = sorted({t["speaker"] for t in turns})
    ref = np.zeros((num_frames, len(speakers)), dtype=bool)
    for t in turns:
        ref[int(round(t["start"] / FRAME)) : int(round(t["end"] / FRAME)), speakers.index(t["speaker"])] = True
    return ref


def best_mapping(ref, hyp):
    """Reference-to-hypothesis speaker pairs maximizing overlap (exhaustive, fine for <= 8 speakers)."""
    overlap = ref.T.astype(np.int64) @ hyp.astype(np.int64)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-overlap)
        return list(zip(rows, cols))
    except ImportError:
        best, best_pairs = -1, []
        for perm in itertools.permutations(range(hyp.shape[1]), min(ref.shape[1], hyp.shape[1])):
            score = sum(overlap[r, c] for r, c in enumerate(perm))
            if score > best:
                best, best_pairs = score, list(enumerate(perm))
        return best_pairs


def der(ref, hyp):
    n_ref, n_hyp = ref.sum(axis=1), hyp.sum(axis=1)
    correct = np.zeros(len(ref), dtype=np.int64)
    for r, c in best_mapping(ref, hyp):
        correct += ref[:, r] & hyp[:, c]
    miss = np.maximum(n_ref - n_hyp, 0).sum()
    false_alarm = np.maximum(n_hyp - n_ref, 0).sum()
    confusion = (np.minimum(n_ref, n_hyp) - correct).sum()
    total = n_ref.sum()
    return (miss + false_alarm + confusion) / total, miss / total, false_alarm / total, confusion / total


def main():
    truth, runs = sys.argv[1], sys.argv[2:]
    for run in runs:
        probs = np.load(run)["probs"]
        hyp = probs > 0.5
        hyp = hyp[:, hyp.any(axis=0)]
        if truth.endswith(".json"):
            ref = truth_matrix(truth, len(probs))
        else:
            ref_probs = np.load(truth)["probs"][: len(probs)]
            ref = ref_probs > 0.5
            ref = ref[:, ref.any(axis=0)]
            hyp = hyp[: len(ref)]
        total, miss, fa, conf = der(ref, hyp)
        print(f"{Path(run).name:40s} DER {total * 100:6.2f}%  (miss {miss * 100:5.2f}, "
              f"false alarm {fa * 100:5.2f}, confusion {conf * 100:5.2f}), {hyp.shape[1]} speakers")


if __name__ == "__main__":
    main()

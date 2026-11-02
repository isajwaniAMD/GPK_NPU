"""Average DER against ground truth over a set of synthetic clips, for several saved runs per clip.

Usage: python eval_set.py cpu npu pipeline_bf16   (reads runs/<clip>.<run>.npz for every audio/*.truth.json)
"""

import sys
from pathlib import Path

import numpy as np

from eval_der import der, truth_matrix

AUDIO, RUNS = Path(__file__).parent / "audio", Path(__file__).parent / "runs"


def main():
    runs = sys.argv[1:]
    clips = sorted(p.name.removesuffix(".truth.json") for p in AUDIO.glob("*.truth.json"))
    totals = {run: [] for run in runs}
    print(f"{'clip':24s}" + "".join(f"{run:>16s}" for run in runs))
    for clip in clips:
        row = f"{clip:24s}"
        for run in runs:
            path = RUNS / f"{clip}.{run}.npz"
            if not path.exists():
                row += f"{'-':>16s}"
                continue
            probs = np.load(path)["probs"]
            hyp = probs > 0.5
            error = der(truth_matrix(AUDIO / f"{clip}.truth.json", len(probs)), hyp[:, hyp.any(axis=0)])[0]
            totals[run].append(error)
            row += f"{error * 100:15.2f}%"
        print(row)
    print(f"{'mean':24s}" + "".join(f"{np.mean(totals[run]) * 100:15.2f}%" if totals[run] else f"{'-':>16s}"
                                    for run in runs))


if __name__ == "__main__":
    main()

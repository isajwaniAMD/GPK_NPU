#!/usr/bin/env python3
"""Sweep diarize_transcribe.py settings on a clip with ground truth and log word error rate and speaker accuracy.

    source ~/ryzenai/env.sh
    python eval/eval_diarization.py multi_speaker_us_en.wav eval/data/multi_speaker_us_en.txt --log eval/results/multi_speaker_accuracy.log

Ground truth lines: "[  start -   end] Speaker N: text". Speaker accuracy is the share of ground-truth speech time
(10 ms frames) assigned to the right speaker under the best one-to-one mapping of predicted to true speakers.
"""
import argparse, itertools, json, re, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (diarize_transcribe.py)
RESULTS = Path(__file__).resolve().parent / "results"
LINE = re.compile(r"\[\s*([\d.]+)\s*-\s*([\d.]+)\]\s*(Speaker \d+):\s*(.*)")


def words(text):
    return re.sub(r"[^\w' ]", " ", text.lower().replace("-", " ")).split()


def edit_distance(r, h):
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)]


def wer(ref, hyp):
    r = words(ref)
    return edit_distance(r, words(hyp)) / max(len(r), 1)


def speaker_accuracy(truth, turns):
    """Frame-level accuracy over true speech under the best predicted->true speaker mapping."""
    pred_labels = sorted({t["speaker_label"] for t in turns})
    true_labels = sorted({t[2] for t in truth})
    frames = []  # (true speaker, {predicted label: frames})
    for start, end, spk, _ in truth:
        counts = {}
        for t in turns:
            ov = min(end, t["end"]) - max(start, t["start"])
            if ov > 0:
                counts[t["speaker_label"]] = counts.get(t["speaker_label"], 0) + int(ov * 100)
        frames.append((spk, counts, int((end - start) * 100)))
    total = sum(f[2] for f in frames)
    best, best_map = 0, {}
    n = max(len(pred_labels), len(true_labels))
    pads = true_labels + [None] * (n - len(true_labels))
    for perm in itertools.permutations(pads, len(pred_labels)):
        mapping = dict(zip(pred_labels, perm))
        ok = sum(c for spk, counts, _ in frames for p, c in counts.items() if mapping[p] == spk)
        if ok > best:
            best, best_map = ok, mapping
    return best / max(total, 1), best_map


def run(audio, out_json, **opts):
    cmd = [sys.executable, str(ROOT / "diarize_transcribe.py"), str(audio), "--json", str(out_json)]
    for k, v in opts.items():
        cmd += [f"--{k}", str(v)]
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])
    return json.loads(Path(out_json).read_text()), time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("truth")
    ap.add_argument("--log", default=str(RESULTS / "accuracy.log"))
    ap.add_argument("--pads", type=float, nargs="+", default=[0.15, 0.3, 0.5, 0.75, 1.0])
    ap.add_argument("--merge-gaps", type=float, nargs="+", default=[0.6])
    ap.add_argument("--min-turns", type=float, nargs="+", default=[0.4])
    a = ap.parse_args()
    truth = [(float(m[1]), float(m[2]), m[3], m[4]) for m in map(LINE.match, Path(a.truth).read_text().splitlines()) if m]
    ref_text = " ".join(t[3] for t in truth)
    lines = [f"Accuracy report: {a.audio} vs {a.truth}", f"Reference: {len(words(ref_text))} words, "
             f"{len(truth)} turns, {len({t[2] for t in truth})} speakers",
             "WER = word errors / reference words (lower is better; punctuation and case ignored, 'back-end' vs "
             "'backend' counts as an error)", ""]
    header = f"{'pad':>5} {'gap':>5} {'min':>5} | {'turns':>5} {'spk':>3} | {'WER %':>6} {'word acc %':>10} | {'speaker acc %':>13} | {'time s':>6}"
    lines += [header, "-" * len(header)]
    results = []
    for pad, gap, mn in itertools.product(a.pads, a.merge_gaps, a.min_turns):
        turns, secs = run(a.audio, RESULTS / "eval_turns.json", **{"pad": pad, "merge-gap": gap, "min-turn": mn})
        hyp = " ".join(t["text"] for t in turns)
        w = wer(ref_text, hyp)
        acc, mapping = speaker_accuracy(truth, turns)
        results.append((w, -acc, pad, gap, mn, turns, hyp, mapping))
        lines.append(f"{pad:5.2f} {gap:5.2f} {mn:5.2f} | {len(turns):5d} {len({t['speaker_label'] for t in turns}):3d} | "
                     f"{w * 100:6.2f} {100 - w * 100:10.2f} | {acc * 100:13.2f} | {secs:6.1f}")
        print(lines[-1], flush=True)
    w, nacc, pad, gap, mn, turns, hyp, mapping = min(results, key=lambda r: (r[0], r[1]))
    lines += ["", f"BEST: pad={pad} merge-gap={gap} min-turn={mn}: WER {w * 100:.2f}%, word accuracy "
              f"{100 - w * 100:.2f}%, speaker accuracy {-nacc * 100:.2f}%", f"Speaker mapping (pred -> true): {mapping}", "",
              "Per-turn detail for the best setting:"]
    for s, e, spk, txt in truth:
        match = max(turns, key=lambda t: min(e, t["end"]) - max(s, t["start"]))
        ok = mapping.get(match["speaker_label"]) == spk
        lines += [f"[{s:5.1f}-{e:5.1f}] {spk} -> {match['speaker_label']} ({'ok' if ok else 'WRONG'}), turn WER "
                  f"{wer(txt, match['text']) * 100:.1f}%", f"   ref: {txt}", f"   hyp: {match['text']}"]
    Path(a.log).write_text("\n".join(lines) + "\n")
    print(f"\nlog written to {a.log}")


if __name__ == "__main__":
    main()

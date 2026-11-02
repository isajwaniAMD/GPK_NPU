#!/usr/bin/env python3
"""Markdown report for an overlapping-speech clip with ground truth.

    source ~/ryzenai/env.sh
    python eval/make_report_overlap.py multi_speaker_overlap_us_en.wav eval/data/multi_speaker_overlap_us_en.txt \
        eval/results/multi_speaker_overlap_accuracy.log eval/results/multi_speaker_overlap.log multi_speaker_overlap_report.md
"""
import json, re, subprocess, sys
from collections import Counter
from pathlib import Path

import numpy as np

from eval_diarization import LINE, RESULTS, ROOT, edit_distance, speaker_accuracy, wer, words

audio, truth_path, sweep_log, whole_log, out_md = sys.argv[1:6]
sweep_text = Path(sweep_log).read_text()
best = re.search(r"BEST: pad=([\d.]+) merge-gap=([\d.]+) min-turn=([\d.]+)", sweep_text)
PAD, GAP, MINT = best.groups()

truth = [(float(m[1]), float(m[2]), m[3], m[4]) for m in map(LINE.match, Path(truth_path).read_text().splitlines()) if m]
run = subprocess.run([sys.executable, str(ROOT / "diarize_transcribe.py"), audio, "--json", str(RESULTS / "report_turns_overlap.json"),
                      "--pad", PAD, "--merge-gap", GAP, "--min-turn", MINT], capture_output=True, text=True, cwd=ROOT, check=True)
timing = next(l for l in run.stdout.splitlines() if l.startswith("Diarization "))
turns = json.loads((RESULTS / "report_turns_overlap.json").read_text())
duration = float(re.search(r"\(([\d.]+) s\)", run.stdout).group(1))
acc, mapping = speaker_accuracy(truth, turns)
speakers = sorted({t[2] for t in truth})
true_of = lambda lab: mapping.get(lab)

ref_all = " ".join(t[3] for t in truth)
hyp_all = " ".join(t["text"] for t in sorted(turns, key=lambda t: t["start"]))
whole_m = re.search(r'Transcription:\n\s+"(.*)"', Path(whole_log).read_text())
whole = whole_m.group(1) if whole_m else ""


def recall(ref_words, hyp_words):
    """Order-independent share of reference words present in the hypothesis (multiset match)."""
    common = Counter(ref_words) & Counter(hyp_words)
    return sum(common.values()) / max(len(ref_words), 1)


# frame-level masks (10 ms)
n = int(duration * 100) + 1
tmask = {s: np.zeros(n, bool) for s in speakers}
for s, e, spk, _ in truth:
    tmask[spk][int(s * 100):int(e * 100)] = True
pmask = {s: np.zeros(n, bool) for s in speakers}
for t in turns:
    spk = true_of(t["speaker_label"])
    if spk:
        pmask[spk][int(t["start"] * 100):int(t["end"] * 100)] = True
active = sum(tmask[s].astype(int) for s in speakers)
overlapped, speech_f = active >= 2, active >= 1
stats = {}
for s in speakers:
    tm, pm = tmask[s], pmask[s]
    ov, cl = tm & overlapped, tm & ~overlapped
    stats[s] = dict(time=tm.sum() / 100, ov_time=ov.sum() / 100, cl_time=cl.sum() / 100,
                    rec=(tm & pm).sum() / max(tm.sum(), 1), rec_ov=(ov & pm).sum() / max(ov.sum(), 1),
                    rec_cl=(cl & pm).sum() / max(cl.sum(), 1), prec=(tm & pm).sum() / max(pm.sum(), 1),
                    pred_time=pm.sum() / 100)

# per true line: matched text = texts of turns labelled with that speaker overlapping the line
lines = []
for s, e, spk, txt in truth:
    ov_s = (overlapped[int(s * 100):int(e * 100)] & tmask[spk][int(s * 100):int(e * 100)]).sum() / 100
    m = [t for t in turns if true_of(t["speaker_label"]) == spk and min(e, t["end"]) - max(s, t["start"]) > 0.3]
    hyp = " ".join(t["text"] for t in sorted(m, key=lambda t: t["start"]))
    lines.append(dict(s=s, e=e, spk=spk, txt=txt, ov=ov_s, frac=ov_s / (e - s), hyp=hyp, rec=recall(words(txt), words(hyp)),
                      span=f"{min(t['start'] for t in m):.1f} - {max(t['end'] for t in m):.1f}" if m else "none"))
clean = [l for l in lines if l["frac"] < 0.10]; over = [l for l in lines if l["frac"] >= 0.10]
group_rec = lambda g: sum(recall(words(l["txt"]), words(l["hyp"])) * len(words(l["txt"])) for l in g) / max(sum(len(words(l["txt"])) for l in g), 1)

speech_s, ov_s_total = speech_f.sum() / 100, overlapped.sum() / 100
L = ["# Multi-speaker overlapping speech, single language (US English): Parakeet TDT 0.6B + Nemotron 3 diarization on the NPU", "",
     "## What was tested", "",
     f"- **Audio:** `{audio}`, {duration:.1f} s, 16 kHz mono. Clean synthetic speech (Piper TTS, US English only), no background noise.",
     f"- **Scenario:** a call-center conversation of {len(truth)} lines by {len(speakers)} speakers: support agent (Speaker 1), customer (Speaker 2), "
     "supervisor (Speaker 3) and a second person on the same account (Speaker 4). People interrupt and talk over each other.",
     f"- **Overlap:** {ov_s_total:.1f} s of the {speech_s:.1f} s of speech ({ov_s_total / speech_s * 100:.0f}%) has two or more people talking at once.",
     f"- **Pipeline and settings:** `diarize_transcribe.py --pad {PAD} --merge-gap {GAP} --min-turn {MINT}` (best of the sweep below).",
     "- **Ground truth:** `multi_speaker_overlap_us_en.txt` (script text and the true start/end of each line; lines overlap in time).",
     f"- **Speed:** {timing}", "",
     "## Overall accuracy", "",
     "| Metric | Value | Meaning |", "|---|---|---|",
     f"| Speakers found / true | {len({t['speaker_label'] for t in turns})} / {len(speakers)} | Distinct speakers the diarizer reported vs the script |",
     f"| Turns found / true | {len(turns)} / {len(truth)} | Detected speaker turns vs scripted lines (overlap makes the diarizer split or merge lines) |",
     f"| Word error rate (ordered) | {wer(ref_all, hyp_all) * 100:.2f}% | Edit distance of all turn texts in time order vs the script. Overlap scrambles order, so this is the strictest view |",
     f"| Word recall (order ignored) | {recall(words(ref_all), words(hyp_all)) * 100:.2f}% | Share of script words that appear anywhere in the output |",
     f"| Word recall, lines with no overlap (<10%) | {group_rec(clean) * 100:.1f}% | {len(clean)} lines |",
     f"| Word recall, lines with overlap (>=10%) | {group_rec(over) * 100:.1f}% | {len(over)} lines |",
     f"| Whole file, no diarization: ordered WER | {wer(ref_all, whole) * 100:.2f}% | Same clip through the Parakeet transcriber in one pass, for comparison |",
     f"| Whole file, no diarization: word recall | {recall(words(ref_all), words(whole)) * 100:.2f}% | |",
     f"| Speaker-time accuracy | {acc * 100:.2f}% | Share of every speaker's speaking time (overlap counted per speaker) given to the right speaker |",
     f"| Label mapping | {', '.join(f'{k} -> {v}' for k, v in mapping.items())} | Predicted label -> true speaker (labels are arbitrary; the diarizer numbers speakers by order of detection) |", "",
     "## Per speaker", "",
     "| Speaker | Role | Speaking time (s) | of which overlapped (s) | Speaking time found (recall) | Recall, no overlap | Recall, overlapped | Precision | Lines | Word recall |",
     "|---|---|---|---|---|---|---|---|---|---|"]
role = {"Speaker 1": "agent", "Speaker 2": "customer", "Speaker 3": "supervisor", "Speaker 4": "2nd caller"}
for s in speakers:
    d = stats[s]; mine = [l for l in lines if l["spk"] == s]
    L.append(f"| {s} | {role.get(s, '')} | {d['time']:.1f} | {d['ov_time']:.1f} ({d['ov_time'] / d['time'] * 100:.0f}%) | {d['rec'] * 100:.1f}% | "
             f"{d['rec_cl'] * 100:.1f}% | {d['rec_ov'] * 100:.1f}% | {d['prec'] * 100:.1f}% | {len(mine)} | {group_rec(mine) * 100:.1f}% |")
L += ["", "Recall = share of the speaker's true speaking time covered by a turn labelled with that speaker. Precision = share of the time the "
      "diarizer labelled as that speaker that really was that speaker. Word recall counts script words found in that speaker's labelled turns.", "",
      "## Per line", "",
      "| # | True speaker | True span (s) | Overlapped | Detected span (s) | Word recall |", "|---|---|---|---|---|---|"]
for i, l in enumerate(lines, 1):
    L.append(f"| {i} | {l['spk']} | {l['s']:.1f} - {l['e']:.1f} | {l['ov']:.1f} s ({l['frac'] * 100:.0f}%) | {l['span']} | {l['rec'] * 100:.0f}% |")
L += ["", "## Transcript differences (lines with missing or changed words)", ""]
for i, l in enumerate(lines, 1):
    if l["rec"] < 1:
        L += [f"- **Line {i} ({l['spk']}, {l['frac'] * 100:.0f}% overlapped, word recall {l['rec'] * 100:.0f}%)**", f"  - reference: {l['txt']}",
              f"  - output:    {l['hyp'] or '(nothing assigned to this speaker)'}"]
sweep = [l for l in sweep_text.splitlines() if re.match(r"\s*\d\.\d\d\s+\d\.\d\d\s+\d\.\d\d \|", l)]
L += ["", "## Parameter sweep", "",
      "| pad (s) | merge gap (s) | min turn (s) | turns | speakers | WER % (ordered) | word acc % | speaker acc % | time (s) |", "|---|---|---|---|---|---|---|---|---|"]
for l in sweep:
    L.append("| " + " | ".join(re.split(r"[\s|]+", l.strip())) + " |")
L += ["", "`--pad` adds audio context before and after each detected turn, `--merge-gap` joins same-speaker segments closer than that, `--min-turn` "
      "drops shorter turns. The sweep ranks settings by ordered WER, then speaker accuracy.", "",
      "## How to read the numbers and limits", "",
      "- **Overlapped speech is a hard case for this pipeline.** Parakeet transcribes one voice per audio piece and cannot separate two people speaking at once, "
      "so words inside overlapped regions are lost or taken from the louder voice. The diarizer assigns the strongest speaker per moment and can only "
      "report simultaneous speakers when it finds separate overlapping segments.",
      "- **Ordered WER is pessimistic here:** overlapped lines arrive as separate turns in start-time order, which differs from the script order. Word recall removes the ordering effect.",
      "- **Speaker-time accuracy** counts overlapped time once for each person speaking, so a perfect result needs both people labelled during an overlap.",
      "- Synthetic voices are acoustically clean and distinct; real call-center audio (telephone band, noise, similar voices) will score lower.",
      "- Reproduce: `python eval_diarization.py ...` (sweep) and `python make_report_overlap.py ...` (this file)."]
Path(out_md).write_text("\n".join(L) + "\n"); print("wrote", out_md)

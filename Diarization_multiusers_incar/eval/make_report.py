#!/usr/bin/env python3
"""Write a markdown report (per-speaker, per-turn and sweep tables) for a clip with ground truth.

    source ~/ryzenai/env.sh
    python eval/make_report.py multi_speaker_us_en.wav eval/data/multi_speaker_us_en.txt multispeaker_singleLingustic.md
"""
import json, re, subprocess, sys
from pathlib import Path

from eval_diarization import LINE, RESULTS, ROOT, edit_distance, speaker_accuracy, wer, words

audio, truth_path, out_md = sys.argv[1:4]
PAD, GAP, MIN_TURN = 0.05, 0.6, 0.4
sweep_log = RESULTS / "multi_speaker_accuracy.log"

truth = [(float(m[1]), float(m[2]), m[3], m[4]) for m in map(LINE.match, Path(truth_path).read_text().splitlines()) if m]
run = subprocess.run([sys.executable, str(ROOT / "diarize_transcribe.py"), audio, "--json", str(RESULTS / "report_turns.json"),
                      "--pad", str(PAD), "--merge-gap", str(GAP), "--min-turn", str(MIN_TURN)],
                     capture_output=True, text=True, cwd=ROOT, check=True)
timing = next(l for l in run.stdout.splitlines() if l.startswith("Diarization "))
turns = json.loads((RESULTS / "report_turns.json").read_text())
acc, mapping = speaker_accuracy(truth, turns)
duration = float(re.search(r"\(([\d.]+) s\)", run.stdout).group(1))

# whole-file Parakeet (no diarization) from the earlier whole-file run
whole = re.search(r'Transcription:\n\s+"(.*)"', (RESULTS / "multi_speaker.log").read_text()).group(1)
ref_all = " ".join(t[3] for t in truth)
hyp_all = " ".join(t["text"] for t in turns)

speakers = sorted({t[2] for t in truth})
rows, per_turn = {s: dict(time=0.0, turns=0, ref=0, err=0, ok_f=0, tot_f=0, pred_t=0.0) for s in speakers}, []
for s, e, spk, txt in truth:
    m = max(turns, key=lambda t: min(e, t["end"]) - max(s, t["start"]))
    ok = mapping.get(m["speaker_label"]) == spk
    r, h = words(txt), words(m["text"])
    err = edit_distance(r, h)
    ov = max(0.0, min(e, m["end"]) - max(s, m["start"]))
    d = rows[spk]
    d["time"] += e - s; d["turns"] += 1; d["ref"] += len(r); d["err"] += err
    d["pred_t"] += m["end"] - m["start"]; d["tot_f"] += (e - s); d["ok_f"] += ov if ok else 0
    per_turn.append((s, e, spk, m, ok, err, len(r), txt, m["text"]))

speech = sum(d["time"] for d in rows.values())
L = []
L += ["# Multi-speaker, single-language (US English) test: Parakeet TDT 0.6B + Nemotron 3 diarization on the NPU", "",
      "## What was tested", "",
      f"- **Audio:** `{audio}`, {duration:.1f} s, 16 kHz mono, clean synthetic speech (Piper TTS, US English only), "
      "no background noise, no overlapping speech, 0.5 s of silence between turns.",
      f"- **Content:** {len(truth)} turns of a project-update meeting by {len(speakers)} speakers, each with a different US English voice "
      "(Speaker 1 = lessac, 2 = ryan, 3 = amy, 4 = joe).",
      "- **Pipeline:** `diarize_transcribe.py` (Nemotron 3 diarization on the NPU, then Parakeet TDT 0.6B BF16 on the NPU per speaker turn, decoder on CPU).",
      f"- **Settings:** `--pad {PAD} --merge-gap {GAP} --min-turn {MIN_TURN}` (best of the sweep below).",
      "- **Ground truth:** `multi_speaker_us_en.txt` (script text and the time each line was synthesized).",
      f"- **Speed:** {timing}", "",
      "## Overall accuracy", "",
      "| Metric | Value | Meaning |", "|---|---|---|",
      f"| Speakers found / true | {len({t['speaker_label'] for t in turns})} / {len(speakers)} | Number of distinct speakers the diarizer reported vs the script |",
      f"| Turns found / true | {len(turns)} / {len(truth)} | Speaker turns detected vs scripted |",
      f"| Word error rate (diarized) | {wer(ref_all, hyp_all) * 100:.2f}% | (substitutions + deletions + insertions) / reference words, punctuation and case ignored |",
      f"| Word accuracy (diarized) | {100 - wer(ref_all, hyp_all) * 100:.2f}% | 100% - WER |",
      f"| Word error rate (whole file, no diarization) | {wer(ref_all, whole) * 100:.2f}% | Same text through the Parakeet transcriber in one pass (4 x 15 s chunks), for comparison |",
      f"| Speaker accuracy | {acc * 100:.2f}% | Share of true speech time (10 ms frames) given to the right speaker, best one-to-one label mapping |",
      f"| Label mapping | {', '.join(f'{k} -> {v}' for k, v in mapping.items())} | Predicted label -> true speaker |", "",
      "## Per speaker", "",
      "| Speaker | Voice | Speaking time (s) | Share of speech | Turns | Reference words | Word errors | WER | Time labelled correctly (s) | Speaker accuracy | Diarized turn time (s) |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
voice = {"Speaker 1": "lessac (F)", "Speaker 2": "ryan (M)", "Speaker 3": "amy (F)", "Speaker 4": "joe (M)"}
for s in speakers:
    d = rows[s]
    L.append(f"| {s} | {voice.get(s, '')} | {d['time']:.1f} | {d['time'] / speech * 100:.1f}% | {d['turns']} | {d['ref']} | {d['err']} | "
             f"{d['err'] / max(d['ref'], 1) * 100:.1f}% | {d['ok_f']:.1f} | {d['ok_f'] / d['tot_f'] * 100:.1f}% | {d['pred_t']:.1f} |")
tot = {k: sum(d[k] for d in rows.values()) for k in ("time", "turns", "ref", "err", "ok_f", "tot_f", "pred_t")}
L.append(f"| **Total** | | **{tot['time']:.1f}** | 100% | **{tot['turns']}** | **{tot['ref']}** | **{tot['err']}** | "
         f"**{tot['err'] / tot['ref'] * 100:.1f}%** | **{tot['ok_f']:.1f}** | **{tot['ok_f'] / tot['tot_f'] * 100:.1f}%** | **{tot['pred_t']:.1f}** |")
L += ["", f"Speech time is {speech:.1f} s of the {duration:.1f} s file; the rest is the 0.5 s silence between turns.", "",
      "## Per turn", "",
      "| # | True speaker | True span (s) | Detected span (s) | Start / end offset (s) | Label | Words | Errors | Turn WER |",
      "|---|---|---|---|---|---|---|---|---|"]
for i, (s, e, spk, m, ok, err, n, txt, hyp) in enumerate(per_turn, 1):
    L.append(f"| {i} | {spk} | {s:.1f} - {e:.1f} | {m['start']:.2f} - {m['end']:.2f} | {m['start'] - s:+.2f} / {m['end'] - e:+.2f} | "
             f"{m['speaker_label']} {'ok' if ok else 'WRONG'} | {n} | {err} | {err / max(n, 1) * 100:.1f}% |")
L += ["", "## Transcript differences", "", "Only turns with word errors (reference vs diarized output):", ""]
diffs = [(i, p) for i, p in enumerate(per_turn, 1) if p[5]]
for i, p in diffs or []:
    L += [f"- **Turn {i} ({p[2]})**", f"  - reference: {p[7]}", f"  - output:    {p[8]}"]
if not diffs:
    L.append("- none")
if sweep_log.exists():
    sweep = [l for l in sweep_log.read_text().splitlines() if re.match(r"\s*\d\.\d\d\s+\d\.\d\d\s+\d\.\d\d \|", l)]
    L += ["", "## Parameter sweep", "",
          "| pad (s) | merge gap (s) | min turn (s) | turns | speakers | WER % | word acc % | speaker acc % | time (s) |", "|---|---|---|---|---|---|---|---|---|"]
    for l in sweep:
        f = re.split(r"[\s|]+", l.strip())
        L.append("| " + " | ".join(f) + " |")
    L += ["", "Each row is a full run of `diarize_transcribe.py` (includes ~3 s model loading). `--pad` is audio context added before and "
          "after each turn; `--merge-gap` merges same-speaker segments closer than that; `--min-turn` drops shorter turns. "
          "Pads of 0.5 s or more pull in the neighbouring speaker's speech. Merge gap and min turn had no effect on this clip because its turns are well separated."]
L += ["", "## How to read the numbers and limits", "",
      "- **WER** counts every word that differs from the script, so formatting differences such as `back-end` vs `backend` or `time out` vs `timeout` count as errors.",
      "- **Speaker accuracy** is computed over true speech time, so a turn that starts 0.3 s late loses 0.3 s. It measures who was assigned when, not only how many turns were right.",
      "- **Offsets** are detected turn start/end minus the true start/end; small positive start offsets and negative end offsets mean the detected turn is slightly shorter than the speech.",
      "- **This is the easy case:** clean synthetic voices, one speaker at a time, one language. Real recordings (noise, overlap, accents, mixed languages) will score lower. "
      "The pad of 0.05 vs 0.15 s differs by a single 0.4 s word in one turn, so treat 0.05-0.15 as roughly equivalent until tested on more audio.",
      "- Reproduce: `python eval_diarization.py ...` (sweep, writes the log) and `python make_report.py ...` (this file)."]
Path(out_md).write_text("\n".join(L) + "\n")
print(f"wrote {out_md}")

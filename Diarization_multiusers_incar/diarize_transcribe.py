#!/usr/bin/env python3
"""
Offline "who said what" pipeline: Nemotron-3 speaker diarization + Parakeet TDT speech-to-text, both on the NPU.

How it works
  1. Diarization (diarization/diarizer.py) splits the audio into speech segments per speaker.
  2. Neighbouring segments of the same speaker are merged into turns.
  3. Parakeet (inference/transcriber.py) transcribes the audio of each turn.
  4. The turns are written out in time order with speaker labels.

Put this file in the root of the parakeetnpu repo (next to the inference/ package) and run it from the Ryzen AI
environment with the same setup as the other scripts:

    source ~/ryzenai/env.sh
    python diarize_transcribe.py diarization/audio/diarization_example.wav --out transcript.txt --json turns.json

Needs the offline diarization graph compiled once (diarization/models/vai_cache/*L704*) and the Parakeet
models prepared (convert_static.py + fuse_pads_direct.py). Audio must be 16 kHz mono 16-bit WAV.
"""

import argparse
import io
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "diarization"))  # for diarizer.py
sys.path.insert(0, str(ROOT))  # for the inference package

SAMPLE_RATE = 16000


def _gap_is_clear(segments, speaker, gap_start, gap_end):
    """True if no other speaker is active anywhere in [gap_start, gap_end]."""
    for s in segments:
        if s["speaker"] != speaker and s["start"] < gap_end and s["end"] > gap_start:
            return False
    return True


def build_turns(segments, merge_gap=0.6, min_dur=0.4):
    """Merge a speaker's nearby segments into turns; drop turns shorter than min_dur seconds.

    Segments of the same speaker are merged when the gap between them is at most merge_gap seconds and no other
    speaker is active in that gap.
    """
    ordered = sorted(segments, key=lambda s: (s["start"], s["speaker"]))
    turns, open_turn = [], {}
    for seg in ordered:
        spk = seg["speaker"]
        cur = open_turn.get(spk)
        if (
            cur is not None
            and seg["start"] - cur["end"] <= merge_gap
            and _gap_is_clear(ordered, spk, cur["end"], seg["start"])
        ):
            cur["end"] = max(cur["end"], seg["end"])
        else:
            cur = {"speaker": spk, "start": seg["start"], "end": seg["end"]}
            open_turn[spk] = cur
            turns.append(cur)
    turns = [t for t in turns if t["end"] - t["start"] >= min_dur]
    turns.sort(key=lambda t: (t["start"], t["speaker"]))
    return turns


def mark_overlaps(turns, fraction=0.3):
    """Flag turns where other speakers talk for at least `fraction` of the turn (their text may mix voices)."""
    for t in turns:
        dur = t["end"] - t["start"]
        overlap = sum(
            max(0.0, min(t["end"], o["end"]) - max(t["start"], o["start"]))
            for o in turns
            if o is not t and o["speaker"] != t["speaker"]
        )
        t["overlap"] = bool(dur > 0 and overlap / dur >= fraction)
    return turns


def to_wav_bytes(samples):
    """16 kHz mono float32 samples -> 16-bit PCM WAV bytes, the format Transcriber.transcribe() takes."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Diarization + Parakeet transcription on the NPU")
    parser.add_argument("audio", help="16 kHz mono 16-bit WAV file")
    parser.add_argument("--asr-device", choices=["cpu", "npu"], default="npu", help="Parakeet encoder device")
    parser.add_argument("--diar-device", choices=["cpu", "npu"], default="npu", help="diarization device")
    parser.add_argument("--models-dir", default=str(ROOT / "models"), help="Parakeet models directory")
    parser.add_argument("--merge-gap", type=float, default=0.6, help="merge same-speaker segments closer than this (s)")
    parser.add_argument("--min-turn", type=float, default=0.4, help="drop turns shorter than this (s)")
    parser.add_argument("--pad", type=float, default=0.15, help="audio context added before/after each turn (s)")
    parser.add_argument("--out", help="write the transcript text to this file")
    parser.add_argument("--json", help="write the turns (start, end, speaker, text) to this JSON file")
    args = parser.parse_args()

    import diarizer as dz
    from inference import Transcriber

    audio = dz.load_audio(args.audio)
    duration = len(audio) / SAMPLE_RATE
    print(f"Audio: {args.audio} ({duration:.1f} s)")

    # ---- load both models (both sessions live in this process and share the NPU) ----
    t0 = time.perf_counter()
    diarizer = dz.Diarizer(args.diar_device, seq_len=704)
    if args.diar_device == "npu":
        for binary in diarizer.npu_binaries():
            if not binary:
                print("WARNING: no compiled NPU binary for the diarizer; it is running on the CPU")
    transcriber = Transcriber(models_dir=args.models_dir, device=args.asr_device, decoder_device="cpu")
    print(f"Models loaded in {time.perf_counter() - t0:.1f} s")

    # ---- 1. diarize ----
    t0 = time.perf_counter()
    probs, mask, _ = diarizer.diarize(audio)
    segments = dz.extract_segments(probs, mask)
    diar_s = time.perf_counter() - t0
    turns = mark_overlaps(build_turns(segments, args.merge_gap, args.min_turn))
    print(f"Diarization: {len(segments)} segments -> {len(turns)} turns in {diar_s:.2f} s "
          f"({duration / diar_s:.0f}x real-time)")

    # ---- 2. transcribe each turn ----
    t0 = time.perf_counter()
    pad = int(args.pad * SAMPLE_RATE)
    lines, results = [], []
    for turn in turns:
        a = max(0, int(turn["start"] * SAMPLE_RATE) - pad)
        b = min(len(audio), int(turn["end"] * SAMPLE_RATE) + pad)
        text = transcriber.transcribe(to_wav_bytes(audio[a:b]), audio_format=".wav").strip()
        if not text:
            continue
        label = f"Speaker {turn['speaker'] + 1}" + (" (overlap)" if turn["overlap"] else "")
        lines.append(f"[{fmt_time(turn['start'])}-{fmt_time(turn['end'])}] {label}: {text}")
        results.append({**turn, "speaker_label": f"Speaker {turn['speaker'] + 1}", "text": text})
    asr_s = time.perf_counter() - t0
    transcriber.close()

    # ---- output ----
    print("\n" + "=" * 70)
    for line in lines:
        print(line)
    print("=" * 70)
    total = diar_s + asr_s
    print(f"Diarization {diar_s:.2f} s + transcription {asr_s:.2f} s = {total:.2f} s "
          f"for {duration:.1f} s of audio ({duration / total:.1f}x real-time, {len(results)} turns)")

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Transcript written to {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"Turns written to {args.json}")


if __name__ == "__main__":
    main()

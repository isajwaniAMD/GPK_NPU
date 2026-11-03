#!/usr/bin/env python3
"""Speaker-aware voice assistant: diarization + Parakeet on the NPU (ONNX Runtime, VitisAI EP), answers from an LLM on
the iGPU (PyTorch, served by llm_server.py).

    # terminal 1 (iGPU venv):   ./start_llm_server.sh
    # terminal 2 (NPU env):     ./run_pipeline.sh incar_commands_us_en.wav

For every speaker turn: transcribe on the NPU, send the text to the LLM with that speaker's own conversation history,
print and log the reply. Needs the Ryzen AI environment (source ~/ryzenai/env.sh) and the compiled diarizer/Parakeet caches.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # the parakeetnpu repo root
sys.path.insert(0, str(ROOT))
import diarize_transcribe as dt         # build_turns, mark_overlaps, to_wav_bytes, fmt_time (also sets up the diarization path)

# SIMULATED car state for the demo (made up; there is no real vehicle or map connection). The assistant may quote these values.
VEHICLE = [
    ("Battery", "62 %, about 185 miles of range"),
    ("Nearest charger", "Electrify America, 4.2 miles away, about 9 minutes, 4 of 8 stalls free, up to 350 kW"),
    ("Charging time", "about 40 minutes from 10 to 80 % at a fast charger, about 9 hours from empty on a home charger"),
    ("Cabin temperature", "front 72 F, rear 76 F (rear zone is adjustable)"),
    ("Music", "playlists available: Relaxing Drive, Road Trip Classics, Acoustic Chill"),
    ("Reminders", "available; a reminder can be set for when the car reaches home"),
]
SYSTEM = ("You are the voice assistant of a car, talking with several occupants. Each message starts with the speaker's label "
          "(Speaker 1, Speaker 2, ...); answer that speaker directly. Reply in one or two short spoken sentences, plain text, "
          "no lists, no markdown. This is a demo with SIMULATED vehicle data; use these values when they are relevant:\n"
          + "\n".join(f"- {k}: {v}" for k, v in VEHICLE) + "\n"
          "For navigation to the nearest charger, use the charger above. For climate, music and reminders, confirm the action as "
          "done and state the new setting. Answer general-knowledge questions directly. Never invent any other vehicle or map "
          "numbers beyond those listed.")


def ask_llm(url, messages, max_new_tokens):
    r = requests.post(f"{url}/chat", json={"messages": messages, "max_new_tokens": max_new_tokens}, timeout=300)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="16 kHz mono 16-bit WAV")
    ap.add_argument("--llm-url", default="http://127.0.0.1:8008")
    ap.add_argument("--max-new-tokens", type=int, default=120)
    ap.add_argument("--history", type=int, default=6, help="exchanges of memory kept per speaker")
    ap.add_argument("--pad", type=float, default=0.05); ap.add_argument("--merge-gap", type=float, default=0.6)
    ap.add_argument("--min-turn", type=float, default=0.4)
    ap.add_argument("--json", default=str(HERE / "asr_llm_results.json")); ap.add_argument("--log", default=str(HERE / "asr_llm_run.log"))
    a = ap.parse_args()

    # the LLM must be on a GPU: refuse to run against a CPU server
    try:
        health = requests.get(f"{a.llm_url}/health", timeout=10).json()
    except requests.RequestException as e:
        sys.exit(f"LLM server not reachable at {a.llm_url} ({e}). Start ./start_llm_server.sh first.")
    if health.get("device") == "cpu":
        sys.exit(f"LLM server is on the CPU, not the iGPU: {health}")

    import diarizer as dz
    from inference import Transcriber

    out, lines = [], []

    def emit(s=""):
        print(s, flush=True); lines.append(s)

    emit(f"LLM server: {health['model']} on {health['device_name']} (torch {health['torch']}, HIP {health['hip']}, {health['dtype']})")
    audio = dz.load_audio(a.audio)
    sr = dt.SAMPLE_RATE
    t0 = time.perf_counter()
    diarizer = dz.Diarizer("npu", seq_len=704)
    transcriber = Transcriber(models_dir=str(ROOT / "models"), device="npu", decoder_device="cpu")
    enc = getattr(getattr(transcriber, "_encoder_session", None), "get_providers", lambda: ["?"])()
    emit(f"ASR models loaded in {time.perf_counter() - t0:.1f}s; Parakeet encoder providers: {enc}")

    t0 = time.perf_counter()
    probs, mask, _ = diarizer.diarize(audio)
    turns = dt.mark_overlaps(dt.build_turns(dz.extract_segments(probs, mask), a.merge_gap, a.min_turn))
    emit(f"Diarization: {len(turns)} turns in {time.perf_counter() - t0:.2f}s\n")

    memory, pad = {}, int(a.pad * sr)
    for turn in turns:
        label = f"Speaker {turn['speaker'] + 1}"
        s, e = max(0, int(turn["start"] * sr) - pad), min(len(audio), int(turn["end"] * sr) + pad)
        t = time.perf_counter()
        text = transcriber.transcribe(dt.to_wav_bytes(audio[s:e]), audio_format=".wav").strip()
        asr_s = time.perf_counter() - t
        if not text:
            continue
        hist = memory.setdefault(label, [])
        messages = [{"role": "system", "content": SYSTEM}] + hist[-2 * a.history:] + [{"role": "user", "content": f"{label}: {text}"}]
        res = ask_llm(a.llm_url, messages, a.max_new_tokens)
        hist += [{"role": "user", "content": f"{label}: {text}"}, {"role": "assistant", "content": res["reply"]}]
        emit(f"[{dt.fmt_time(turn['start'])}-{dt.fmt_time(turn['end'])}] {label} (NPU ASR {asr_s * 1000:.0f} ms): {text}")
        emit(f"    Assistant -> {label} (iGPU, {res['new_tokens']} tokens in {res['seconds']:.2f}s, GPU busy peak {res['gpu_busy_max']}%): {res['reply']}\n")
        out.append({"start": turn["start"], "end": turn["end"], "speaker": label, "overlap": turn["overlap"], "heard": text,
                    "reply": res["reply"], "asr_seconds": round(asr_s, 3), "llm_seconds": res["seconds"],
                    "llm_new_tokens": res["new_tokens"], "gpu_busy_max": res["gpu_busy_max"]})
    transcriber.close()
    Path(a.json).write_text(json.dumps({"llm": health, "turns": out}, indent=1, ensure_ascii=False), encoding="utf-8")
    Path(a.log).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"results: {a.json}\nlog: {a.log}")


if __name__ == "__main__":
    main()

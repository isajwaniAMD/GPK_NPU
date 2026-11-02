#!/usr/bin/env python3
"""Browser GUI demo: speaker-aware in-car voice assistant.

Plays a multi-speaker WAV in the browser while the server runs the same pipeline as
ASR_LLM_inference/asr_llm_pipeline.py: diarization (NPU) -> Parakeet ASR (NPU encoder, CPU decoder) -> Qwen3 LLM (iGPU),
with a separate conversation memory per speaker. Every event carries its turn_id and speaker so the page can show
each reply under the turn that asked it.

    ./run_demo.sh [wav]          # starts the LLM server if needed, then this app on http://127.0.0.1:8080

Needs the Ryzen AI environment (source ~/ryzenai/env.sh) and the LLM server on :8008 (ASR_LLM_inference/start_llm_server.sh).
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "ASR_LLM_inference"))  # reuse SYSTEM prompt and ask_llm, read-only
sys.path.insert(0, str(ROOT))
import asr_llm_pipeline as pipe  # noqa: E402
import diarize_transcribe as dt  # noqa: E402

app = FastAPI(title="Speaker-aware voice assistant demo")
CFG = {"audio": ROOT / "ASR_LLM_inference" / "incar_commands_us_en.wav", "llm_url": "http://127.0.0.1:8008",
       "history": 6, "max_new_tokens": 120, "pad": 0.05, "merge_gap": 0.6, "min_turn": 0.4}

MODELS = {}                      # diarizer, transcriber, dz (loaded once)
RUN = {"events": [], "running": False, "id": 0, "samples": []}   # samples: (cpu, npu, igpu) per usage sample during a run
COND = threading.Condition()


def emit(**event):
    with COND:
        RUN["events"].append(event)
        COND.notify_all()


# Device usage shown in the page legend. CPU comes from /proc/stat and the iGPU from the kernel's gpu_busy_percent (both
# system-wide). The NPU driver exposes no utilization counter, so NPU is the share of each sample window during which one of
# our NPU inferences (diarizer step graph, Parakeet encoder) was executing.
USAGE = {"cpu": 0.0, "npu": 0.0, "igpu": None, "t": 0.0}
NPU = {"busy": [], "active": None, "lock": threading.Lock()}
IGPU_FILE = next(iter(sorted(Path("/sys/class/drm").glob("card*/device/gpu_busy_percent"))), None)
SAMPLE_S = 0.5


def track_npu(session):
    """Time every session.run so the sampler knows when the NPU was busy."""
    run = session.run

    def timed(*args, **kwargs):
        start = time.monotonic()
        with NPU["lock"]:
            NPU["active"] = start
        try:
            return run(*args, **kwargs)
        finally:
            with NPU["lock"]:
                NPU["busy"].append((start, time.monotonic()))
                NPU["active"] = None
    session.run = timed


def read_cpu():
    fields = [int(x) for x in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
    return sum(fields), fields[3] + fields[4]  # total, idle + iowait


def sample_usage():
    prev_total, prev_idle = read_cpu()
    prev = time.monotonic()
    while True:
        time.sleep(SAMPLE_S)
        now = time.monotonic()
        total, idle = read_cpu()
        cpu = 100.0 * (1.0 - (idle - prev_idle) / max(1, total - prev_total))
        with NPU["lock"]:
            spans = NPU["busy"] + ([(NPU["active"], now)] if NPU["active"] is not None else [])
            busy = sum(max(0.0, min(e, now) - max(s, prev)) for s, e in spans)
            NPU["busy"] = [(s, e) for s, e in NPU["busy"] if e > now]
        igpu = None
        if IGPU_FILE is not None:
            try:
                igpu = float(IGPU_FILE.read_text())
            except (OSError, ValueError):
                pass
        USAGE.update(cpu=round(max(0.0, cpu), 1), npu=round(min(100.0, 100.0 * busy / (now - prev)), 1), igpu=igpu, t=time.time())
        if RUN["running"]:
            RUN["samples"].append((USAGE["cpu"], USAGE["npu"], USAGE["igpu"]))
        prev, prev_total, prev_idle = now, total, idle


def usage_summary():
    """Average and peak usage of each device over the whole run, for the page's end-of-run summary."""
    rows = list(RUN["samples"])
    out = {"seconds": round(len(rows) * SAMPLE_S, 1)}
    for i, name in enumerate(("cpu", "npu", "igpu")):
        values = [r[i] for r in rows if r[i] is not None]
        out[name] = {"avg": round(sum(values) / len(values), 1), "peak": round(max(values), 1)} if values else None
    if out["npu"]:
        out["npu"]["busy_s"] = round(out["npu"]["avg"] / 100.0 * out["seconds"], 1)
    return out


def load_models():
    import diarizer as dz
    from inference import Transcriber
    t = time.perf_counter()
    MODELS["dz"] = dz
    MODELS["diarizer"] = dz.Diarizer("npu", seq_len=704)
    MODELS["transcriber"] = Transcriber(models_dir=str(ROOT / "models"), device="npu", decoder_device="cpu")
    for session in MODELS["diarizer"].sessions + [MODELS["transcriber"]._encoder_session]:
        track_npu(session)
    MODELS["load_s"] = round(time.perf_counter() - t, 1)


def run_pipeline(run_id):
    try:
        dz, diarizer, transcriber = MODELS["dz"], MODELS["diarizer"], MODELS["transcriber"]
        t0 = time.monotonic()  # the browser starts playback on click, so the playback clock starts now
        audio, sr = dz.load_audio(str(CFG["audio"])), dt.SAMPLE_RATE
        t = time.perf_counter()
        probs, mask, _ = diarizer.diarize(audio)
        turns = dt.mark_overlaps(dt.build_turns(dz.extract_segments(probs, mask), CFG["merge_gap"], CFG["min_turn"]))
        emit(type="diarized", seconds=round(time.perf_counter() - t, 2), turns=len(turns), duration=round(len(audio) / sr, 1))

        memory, pad = {}, int(CFG["pad"] * sr)
        for i, turn in enumerate(turns):
            label = f"Speaker {turn['speaker'] + 1}"
            # Reveal each turn when playback reaches its end, as a live system would hear it.
            time.sleep(max(0.0, t0 + turn["end"] - time.monotonic()))
            emit(type="turn", id=i, speaker=label, spk=turn["speaker"], start=turn["start"], end=turn["end"],
                 overlap=turn["overlap"])
            s, e = max(0, int(turn["start"] * sr) - pad), min(len(audio), int(turn["end"] * sr) + pad)
            t = time.perf_counter()
            text = transcriber.transcribe(dt.to_wav_bytes(audio[s:e]), audio_format=".wav").strip()
            asr_ms = round((time.perf_counter() - t) * 1000)
            emit(type="transcript", id=i, speaker=label, text=text, asr_ms=asr_ms)
            if not text:
                continue
            hist = memory.setdefault(label, [])
            user = {"role": "user", "content": f"{label}: {text}"}
            messages = [{"role": "system", "content": pipe.SYSTEM}] + hist[-2 * CFG["history"]:] + [user]
            res = pipe.ask_llm(CFG["llm_url"], messages, CFG["max_new_tokens"])
            hist += [user, {"role": "assistant", "content": res["reply"]}]
            emit(type="reply", id=i, speaker=label, reply=res["reply"], tokens=res["new_tokens"],
                 llm_s=round(res["seconds"], 2), gpu_busy=res["gpu_busy_max"], memory=len(hist) // 2)
        emit(type="done", total_s=round(time.monotonic() - t0, 1), usage=usage_summary())
    except Exception as exc:  # surface to the page instead of dying silently
        emit(type="error", message=f"{type(exc).__name__}: {exc}")
    finally:
        with COND:
            RUN["running"] = False


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "index.html").read_text(encoding="utf-8")


@app.get("/audio.wav")
def audio():
    return FileResponse(CFG["audio"], media_type="audio/wav")


@app.get("/api/status")
def status():
    try:
        llm = requests.get(f"{CFG['llm_url']}/health", timeout=2).json()
    except requests.RequestException:
        llm = None
    return {"models_loaded": "transcriber" in MODELS, "load_s": MODELS.get("load_s"), "running": RUN["running"],
            "llm": llm, "audio": CFG["audio"].name}


@app.get("/api/usage")
def usage():
    return USAGE


@app.post("/api/start")
def start():
    with COND:
        if RUN["running"]:
            raise HTTPException(409, "a run is already in progress")
        RUN.update(running=True, events=[], samples=[], id=RUN["id"] + 1)
        run_id = RUN["id"]
    threading.Thread(target=run_pipeline, args=(run_id,), daemon=True).start()
    return {"run": run_id}


@app.get("/api/events")
def events():
    def stream():
        n, run_id = 0, RUN["id"]
        while True:
            with COND:
                COND.wait_for(lambda: len(RUN["events"]) > n or RUN["id"] != run_id, timeout=15)
                if RUN["id"] != run_id:
                    return
                batch, n = RUN["events"][n:], len(RUN["events"])
            if not batch:
                yield ": keepalive\n\n"
            for ev in batch:
                yield f"data: {json.dumps(ev)}\n\n"
                if ev["type"] in ("done", "error"):
                    return
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="?", default=str(CFG["audio"]), help="16 kHz mono 16-bit WAV")
    ap.add_argument("--llm-url", default=CFG["llm_url"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    CFG.update(audio=Path(a.audio).resolve(), llm_url=a.llm_url)
    print("loading diarizer + Parakeet on the NPU ...", flush=True)
    load_models()
    threading.Thread(target=sample_usage, daemon=True).start()
    print(f"models ready in {MODELS['load_s']}s; open http://{a.host}:{a.port}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()

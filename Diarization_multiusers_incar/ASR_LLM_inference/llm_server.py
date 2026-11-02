#!/usr/bin/env python3
"""Local LLM server on the iGPU through PyTorch (ROCm/HIP). Stateless chat endpoint, stdlib HTTP only.

    source .venv-igpu/bin/activate
    python llm_server.py --model ../models/Qwen3-4B-Instruct-2507 --port 8008

GET  /health  -> device, torch/HIP versions, model, memory
POST /chat    -> {"messages": [{"role": "...", "content": "..."}], "max_new_tokens": 120}
              <- {"reply": "...", "prompt_tokens": n, "new_tokens": n, "seconds": s, "gpu_busy_max": pct}

The server refuses to start on the CPU unless --allow-cpu is given, so a run can never silently leave the iGPU.
"""
import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BUSY = Path("/sys/class/drm/card0/device/gpu_busy_percent")


def gpu_busy():
    try:
        return int(BUSY.read_text())
    except (OSError, ValueError):
        return -1


class Sampler(threading.Thread):
    """Samples the amdgpu busy percentage while a reply is generated (evidence that the iGPU did the work)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop, self.peak = threading.Event(), -1

    def run(self):
        while not self.stop.is_set():
            self.peak = max(self.peak, gpu_busy())
            time.sleep(0.05)


class Engine:
    def __init__(self, model_id, dtype, device):
        self.device, self.model_id, self.lock = device, model_id, threading.Lock()
        self.dtype = getattr(torch, dtype)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.dtype).to(device).eval()

    def chat(self, messages, max_new_tokens):
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        with self.lock:
            sampler = Sampler(); sampler.start()
            t = time.perf_counter()
            with torch.inference_mode():
                out = self.model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
            if self.device != "cpu":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - t
            sampler.stop.set(); sampler.join()
        new = out[0, ids["input_ids"].shape[1]:]
        reply = re.sub(r"<think>.*?</think>", "", self.tok.decode(new, skip_special_tokens=True), flags=re.S).strip()
        return {"reply": reply, "prompt_tokens": int(ids["input_ids"].shape[1]), "new_tokens": int(len(new)),
                "seconds": round(seconds, 3), "gpu_busy_max": sampler.peak}

    def health(self):
        on_gpu = self.device != "cpu"
        return {"model": self.model_id, "device": self.device, "dtype": str(self.dtype),
                "device_name": torch.cuda.get_device_name(0) if on_gpu else "cpu",
                "torch": torch.__version__, "hip": torch.version.hip,
                "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 2) if on_gpu else 0.0}


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            self._send(200, engine.health()) if self.path == "/health" else self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/chat":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, engine.chat(req["messages"], int(req.get("max_new_tokens", 120))))
            except Exception as e:  # report to the client instead of dropping the connection
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *a):
            pass
    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Hugging Face repo id or local folder")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--allow-cpu", action="store_true", help="debug only: run on the CPU if no GPU is visible")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        if not a.allow_cpu:
            raise SystemExit(f"No GPU visible to PyTorch (torch {torch.__version__}, hip {torch.version.hip}). "
                             "Check ROCm torch is installed and that you are in the 'render' group (/dev/kfd).")
        device = "cpu"
    else:
        device = "cuda"
    print(f"loading {a.model} on {device} ({a.dtype}) ...", flush=True)
    t = time.perf_counter()
    engine = Engine(a.model, a.dtype, device)
    print(f"loaded in {time.perf_counter() - t:.1f}s: {engine.health()}", flush=True)
    print(f"serving on http://{a.host}:{a.port}", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(engine)).serve_forever()


if __name__ == "__main__":
    main()

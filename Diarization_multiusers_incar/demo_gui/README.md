# Speaker-aware voice assistant: browser demo

Plays `ASR_LLM_inference/incar_commands_us_en.wav` (4 speakers, 10 questions) while the pipeline runs and shows each
answer under the question that asked it. Diarization and Parakeet use the NPU, Qwen3-4B uses the iGPU.

## Run

```bash
~/isdev/NPU_WL/parakeetnpu_multizone_diarization/demo_gui/run_demo.sh
```

It starts the LLM server (iGPU) if it is not already up, then the app. Wait for `models ready ... open http://127.0.0.1:8080`
(about 15 s), then open http://127.0.0.1:8080 in Chrome and press **Start demo**.

Over SSH, forward the port from your laptop first: `ssh -L 8080:127.0.0.1:8080 isajwani@<host>`, then browse to
http://127.0.0.1:8080 on the laptop.

## Stop

`Ctrl+C` in the terminal running `run_demo.sh` stops the app. The LLM server keeps running in the background
(log: `demo_gui/llm_server.log`); stop it with `pkill -f llm_server.py`.

## Files

- `app.py`: FastAPI backend (runs the pipeline, streams events to the page, serves `/api/usage`)
- `index.html`: the page (no external libraries)
- `run_demo.sh`: starts everything

## Compute usage panel

The left column shows live NPU / iGPU / CPU usage (updated every 0.5 s, 30 s sparkline). CPU is system-wide from `/proc/stat`,
the iGPU is the kernel's `gpu_busy_percent` (system-wide, so it includes the LLM server), and the NPU is the share of each
sample window in which one of the demo's NPU inferences (diarizer step graph, Parakeet encoder) was running. The NPU driver
exposes no utilization counter, so that value reflects this demo's own NPU work, not other processes.

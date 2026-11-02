# ASR on the NPU, LLM on the iGPU: speaker-aware voice assistant

```
incar_commands_us_en.wav ──► diarization + Parakeet TDT   (ONNX Runtime, VitisAI EP, NPU)
                                   │ one text per speaker turn
                                   ▼
                          asr_llm_pipeline.py ── HTTP localhost:8008 ──► llm_server.py (PyTorch ROCm, iGPU)
                          per-speaker memory  ◄────── reply ────────────  Qwen3-4B-Instruct (fp16)
```

Two processes on purpose: the NPU side uses the Ryzen AI libraries (`~/ryzenai/env.sh`), the LLM side uses its own ROCm
PyTorch venv (`.venv-igpu`), so their native libraries never mix.

## Files
| File | Role |
|---|---|
| `make_incar_wav.py` | Makes the test clip (4 US English voices, 10 questions to the car assistant, no noise, no overlap) + ground truth `incar_commands_us_en.txt` |
| `incar_commands_us_en.wav` / `.txt` | The clip (42.2 s, 16 kHz mono) and its ground truth |
| `llm_server.py`, `start_llm_server.sh` | LLM on the iGPU through PyTorch; refuses to start on the CPU |
| `asr_llm_pipeline.py`, `run_pipeline.sh` | NPU ASR per speaker turn, then the LLM, per-speaker memory; writes `asr_llm_results.json` and `asr_llm_run.log` |
| `check_igpu.py` | Confirms PyTorch runs on the Radeon 8060S (gfx1151) |
| `download_model.sh` | Downloads the model with curl |
| `.venv-igpu/` | ROCm PyTorch 2.12 (ROCm 7.13, gfx1151 nightly index), transformers, accelerate |

## One-time setup (needs you: sudo and network)
```bash
# 1. let your user open the GPU (/dev/kfd is root:render); then LOG OUT AND BACK IN
sudo usermod -aG render,video $USER
id -nG                                   # must list render and video

# 2. check PyTorch sees the iGPU (run from this folder; create .venv-igpu first, see the repo's top-level README)
cd ASR_LLM_inference                     # from the repo root
.venv-igpu/bin/python check_igpu.py

# 3. download the model (about 8 GB)
./download_model.sh
```

## Run
```bash
# terminal 1: LLM on the iGPU
./start_llm_server.sh                   # uses ../models/Qwen3-4B-Instruct-2507

# terminal 2: ASR on the NPU, answers from the LLM
./run_pipeline.sh                       # uses incar_commands_us_en.wav
```

Notes: the nightly ROCm wheels come from `https://rocm.nightlies.amd.com/v2/gfx1151/`; nightlies can change, and
bf16 problems are reported on gfx1151, so the server defaults to fp16 (`--dtype`). Do not set `PYTORCH_HIP_ALLOC_CONF=backend:malloc`.

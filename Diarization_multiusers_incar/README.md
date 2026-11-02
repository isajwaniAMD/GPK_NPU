# Speaker-aware voice assistant on AMD Ryzen AI (Strix Halo)

Several people talk in a car; the assistant works out **who** is speaking, transcribes each turn, and answers each speaker
by name, remembering what *that* speaker asked before. Everything runs on one machine, nothing goes to the cloud.

```
audio ──► Nemotron-3 diarization ──► speaker turns ──► Parakeet TDT speech-to-text ──► Qwen3-4B LLM ──► reply to that speaker
          (NPU)                      id + speaker      (encoder NPU, decoder CPU)       (iGPU, PyTorch ROCm)
```

| Stage | Runs on |
|---|---|
| Diarization network (`diar_step_L704.onnx`, VitisAI EP, BF16) | NPU |
| Diarization front end, speaker cache, turn building | CPU (numpy) |
| Parakeet mel features | CPU (numpy) |
| Parakeet encoder (static, Pad-fused ONNX, VitisAI EP, BF16) | NPU |
| Parakeet decoder/joint (TDT) | CPU |
| Qwen3-4B-Instruct-2507 (fp16) | iGPU (Radeon 8060S, PyTorch ROCm) |

The NPU side and the LLM side are separate processes on purpose (Ryzen AI libraries vs. ROCm PyTorch), talking over HTTP on
`localhost:8008`.

## How each reply finds the right speaker
- Diarization produces turns `{id, speaker, start, end}`; only that turn's audio is transcribed, so a transcript never mixes voices.
- The transcript is sent to the LLM as `"Speaker N: <text>"` together with **only that speaker's** previous exchanges
  (a separate memory per speaker label, last 6 exchanges by default).
- The reply is produced in the same step as its question, and the browser demo tags every event with the turn `id` and speaker
  and places the reply under the card with the same `id`.
- Scope today: clean speech, one speaker at a time. Overlapping turns are flagged (`overlap`) but not handled.

## Repository layout
```
inference/                   Parakeet TDT engine (from inference import Transcriber)
  transcriber.py             ONNX Runtime pipeline: NPU encoder, CPU/iGPU decoder
  mel.py, audio.py           128-bin mel features, WAV parsing
  tools/                     model preparation: download_models, convert_static, fuse_pads_direct, optimize_model, fuse_attn_pads
diarization/                 Nemotron-3 diarization on the NPU (see diarization/README.md)
diarize_transcribe.py        diarize + transcribe a WAV into labelled turns (also imported by ASR_LLM_inference/ and demo_gui/)
ASR_LLM_inference/           the assistant pipeline: asr_llm_pipeline.py + llm_server.py, the 4-speaker test clip (see its README)
demo_gui/                    browser demo (FastAPI + one HTML page), see demo_gui/README.md
eval/                        diarization/WER evaluation scripts, ground truth (data/) and saved results (results/)
audio_files/                 multi_speaker_us_en.wav, the clip the eval scripts use
models/                      model weights (not in git); small config files are tracked
fetch_models.sh              downloads / builds every model
requirements.txt             Python packages for the Ryzen AI side
```

## Requirements
- AMD Ryzen AI machine (developed on Strix Halo, Radeon 8060S / gfx1151), tested on Linux.
- **Ryzen AI environment** with `onnxruntime` + VitisAI EP, activated with `source ~/ryzenai/env.sh`
  (Ryzen AI SDK 1.8.0 for the BF16 encoder; 1.7.x failed to compile it). Install `pip install -r requirements.txt` into it.
  Do **not** `pip install onnxruntime` there, it would replace the VitisAI build.
- **iGPU environment** for the LLM: a separate venv, `ASR_LLM_inference/.venv-igpu`, with ROCm PyTorch from the gfx1151 nightly
  index plus `transformers` and `accelerate`:
  ```bash
  cd ASR_LLM_inference
  uv venv .venv-igpu --python 3.12
  uv pip install --python .venv-igpu/bin/python torch --index-url https://rocm.nightlies.amd.com/v2/gfx1151/
  uv pip install --python .venv-igpu/bin/python transformers accelerate
  sudo usermod -aG render,video $USER     # then log out and back in; /dev/kfd must be accessible
  .venv-igpu/bin/python check_igpu.py     # confirms PyTorch sees the iGPU
  ```
  (Tested with torch 2.12.0a0+rocm7.13.0a20260411. Nightly wheels change; the server defaults to fp16 because bf16 problems
  are reported on gfx1151. Do not set `PYTORCH_HIP_ALLOC_CONF=backend:malloc`.)

## Models
Weights are not in git. Fetch or build everything with:
```bash
source ~/ryzenai/env.sh
./fetch_models.sh            # all; or: llm | asr | diar   (add --check to only list what is missing)
```
| What | Where | How |
|---|---|---|
| Qwen3-4B-Instruct-2507 (~8 GB) | `models/Qwen3-4B-Instruct-2507/` | downloaded |
| Parakeet NPU encoder + static decoder | `models/encoder-model.fp32.static.npu.onnx(+.data)`, `models/decoder_joint-model.fp32.static.onnx` | downloaded, then `convert_static` + `fuse_pads_direct` |
| Diarizer step graph | `diarization/models/diar_step_L704.onnx`, `diar_aux.npz` | exported (needs the separate `nemo-diar-export` conda env, see `diarization/README.md`) |
| Small configs (`config.json`, `static_config.json`, `vocab.txt`, `vai_ep_config*.json`) | `models/`, `diarization/models/` | tracked in git |

The compiled NPU binaries (`models/vai_cache/`, `diarization/models/vai_cache/`) are built on the first run (the diarizer takes
35-60 minutes, the encoder about 30) and are specific to your driver and SDK version; do not share them.

## Run
**Browser demo** (starts the LLM server if needed, then the app):
```bash
demo_gui/run_demo.sh            # then open http://127.0.0.1:8080 and press Start demo
```
**Command line pipeline:**
```bash
ASR_LLM_inference/start_llm_server.sh      # terminal 1: LLM on the iGPU (port 8008)
ASR_LLM_inference/run_pipeline.sh          # terminal 2: diarize + transcribe on the NPU, answers from the LLM
```
**Diarize + transcribe only (no LLM):**
```bash
source ~/ryzenai/env.sh
python diarize_transcribe.py ASR_LLM_inference/incar_commands_us_en.wav --out transcript.txt --json turns.json
```
**Evaluation** (word error rate and speaker accuracy on the 4-speaker clip):
```bash
python eval/eval_diarization.py audio_files/multi_speaker_us_en.wav eval/data/multi_speaker_us_en.txt --log eval/results/accuracy.log
```

## Reference run (42 s clip, 4 speakers, 10 questions)
Diarization about 0.6-0.8 s for the whole clip; speech-to-text about 0.55 s per turn on the NPU; answers 0.5-2.4 s per turn on
the iGPU; all 10 questions transcribed correctly and answered to the right speaker. The browser demo replays the clip in real time.

## Troubleshooting
- **`run_pipeline.sh` exits silently:** the script uses `set -u` and the Ryzen AI venv's `activate` reads `LD_LIBRARY_PATH` and
  `PYTHONPATH`. Run `LD_LIBRARY_PATH= PYTHONPATH= ASR_LLM_inference/run_pipeline.sh`.
- **Port 8008 already in use / demo talks to the wrong LLM server:** `run_demo.sh` skips starting a server when `/health`
  answers. Stop the old one with `pkill -f llm_server.py`.
- **`vaiml.dll not found` warning:** harmless on Linux when the encoder still reports `VitisAIExecutionProvider`.
- **First NPU run is very slow:** it is compiling the NPU binary (see Models). Later runs load the cache in a few seconds.
- **LLM server refuses to start / is on the CPU:** check group membership (`id -nG` must list `render` and `video`) and `check_igpu.py`.
- **Audio format:** the pipeline needs 16 kHz mono 16-bit WAV: `ffmpeg -i in.mp3 -ar 16000 -ac 1 -c:a pcm_s16le out.wav`.

## Credits
NVIDIA (Parakeet TDT 0.6B, Nemotron 3 Diarization), Ivan Stupakov (@istupakov, Parakeet ONNX conversion), achetronic (original Go
mel/decoder implementation), Alibaba Qwen (Qwen3-4B-Instruct-2507), AMD (Ryzen AI NPU, VitisAI EP, ROCm).

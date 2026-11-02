# Nemotron 3 Diarization on the AMD Ryzen AI NPU

Part of the Parakeet NPU repo ([main README](../README.md)). Speaker diarization ("who spoke when") with [NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)
running on the Ryzen AI NPU (Strix, XDNA2) through the VitisAI Execution Provider in BF16, with Ryzen AI SDK 1.8.0.

- **Offline** (whole files): 16.5 min of audio in 5.6 s on the NPU (175x real time, NPU in turbo mode), vs 28-32x
  on the CPU.
- **Streaming** (live, 1.04 s buffer): each 0.72 s step takes 160-210 ms on the NPU, well inside the real-time budget.
## How it works

The model is a 31-layer Transformer (100M parameters) run chunk by chunk, each chunk attending to an
Arrival-Order Speaker Cache and a FIFO of earlier frames. The work is split as follows:

| Part | Where | Notes |
|---|---|---|
| Log-mel features, 8-frame stacking projection | CPU (numpy) | Computed in a background thread, overlapped with the NPU |
| Transformer, projection, upsampler, classifier (one chunk step) | NPU | One static-shape ONNX graph, 100% of its operators on the NPU |
| Speaker cache / FIFO update and compression | CPU (numpy) | Line-by-line port of the Transformers implementation |

Each step's input (cache + FIFO + chunk + look-ahead) varies in length, so it is padded to a fixed size and the
padding is masked with a finite float attention bias (no BOOL tensors) and zeroed before the upsampler convolution.
Valid frames match an unpadded forward exactly. Two step graphs are used:

| Graph | Frames | Operating point (model card) |
|---|---|---|
| `diar_step_L704.onnx` | 704 (684 needed) | Offline: chunk 340, look-ahead 40, FIFO 40, cache 264, 30.4 s buffer |
| `diar_step_L544.onnx` | 544 (541 needed) | Low latency: chunk 9, look-ahead 4, FIFO 264, cache 264, 1.04 s buffer |

On the CPU both pipelines reproduce the official Transformers model exactly (offline and streaming modes: 100%
identical speaker decisions).

## Results

Measured on a 16.5-minute TED talk (`../ted.wav`) and on 10 multi-speaker clips (see [Test data](#test-data)).

### Speed (offline)

| | NPU default power mode | NPU turbo mode | CPU (ONNX Runtime FP32, 8 threads) |
|---|---|---|---|
| 16.5 min TED talk | 9.1 s (109x) | 5.6 s (175x) | 30.6-35.2 s (28-32x) |
| Per 27 s chunk step | 240 ms | 146 ms | ~800 ms |

PyTorch (Transformers, CPU) runs at about 30x. The NPU step is about 98% of the NPU runtime.

### Accuracy

DER is frame-level, overlap scored, no collar, with the best speaker mapping. The absolute numbers on the synthetic
clips are inflated because pauses inside each source recording count as speech in the ground truth, so compare
rows, not absolute values.

| Run | Mean DER vs ground truth, 9 synthetic clips | DER vs FP32 reference, real 6-speaker clip |
|---|---|---|
| CPU FP32 (= PyTorch reference) | 16.34% | 0% |
| PyTorch BF16 | 16.37% | 0.18% |
| NPU, `vai_ep_config.json` | 17.53% | 5.49% |
| **NPU, `vai_ep_config_accurate.json` (default)** | **17.01%** | **0.16%** |
| NPU streaming, low latency (1.04 s) | 15.95% | 8.04% vs the streaming reference |

The speaker cache amplifies small numeric differences: one borderline cache decision can split a speaker across two
channels and move a clip's DER by several points, so single-clip comparisons are noisy.

## Setup

Two conda environments:

- **`ryzen-ai-1.8.0`** (Ryzen AI SDK 1.8.0): compiling and running on the NPU. The only additions were
  `sounddevice` (with `cffi`, `pycparser`) for live microphone input.
- **`nemo-diar-export`**: only for exporting the model and building PyTorch references. It needs the Transformers
  development build, which conflicts with the pinned packages in the Ryzen AI env. Do not install it into
  `ryzen-ai-1.8.0` (it would also replace `onnxruntime-vitisai` with plain `onnxruntime`).

```powershell
conda create -y -n nemo-diar-export python=3.12
conda activate nemo-diar-export
python -m pip install -r requirements-export.txt
```

Download the model (with `curl.exe`, which uses the Windows certificate store, where the Hugging Face client fails
behind SSL inspection):

```powershell
$d = "models\hf"; New-Item -ItemType Directory -Force $d | Out-Null
foreach ($f in 'config.json','processor_config.json','model.safetensors') {
    curl.exe -sSL -o "$d\$f" "https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/main/$f"
}
```

Export the step graphs (writes `models/diar_step_L*.onnx` and `models/diar_aux.npz`, the CPU-side weights):

```powershell
conda activate nemo-diar-export
python export_onnx.py --seq-len 704   # offline
python export_onnx.py --seq-len 544   # streaming
```

The first NPU run of each graph compiles it with VAIML into `models/vai_cache/<key>/<key>.rai` (126-132 MB). This
takes 35-60 minutes and must not be interrupted; later runs load the cached binary in about 4 s. To compile without
tying up a terminal:

```powershell
conda activate ryzen-ai-1.8.0
python -c "import diarizer as dz; dz.Diarizer('npu', seq_len=704)"
python -c "import diarizer as dz; dz.Diarizer('npu', seq_len=544)"
```

## Usage

All commands run in `ryzen-ai-1.8.0` from this folder. Audio must be 16 kHz mono; without `soundfile` installed,
16-bit PCM WAV is required (convert with `ffmpeg -i in.m4a -ar 16000 -ac 1 -c:a pcm_s16le out.wav`).

```powershell
# offline diarization of a file
python diarizer.py meeting.wav --device npu --json segments.json
python diarizer.py meeting.wav --device cpu          # full-precision comparison

# a file fed as a live stream (100 ms blocks), low-latency mode
python stream_diarizer.py meeting.wav --device npu

```

`diarizer.py` prints the providers, the NPU binary in use, timings per stage and the first segments. If the
compiled binary is missing it prints `WARNING: no compiled NPU binary ... running on the CPU`, so a silent CPU
fallback is visible. Options: `--runs N` (best of N warm runs), `--save out.npz` (probabilities for scoring),
`--vai-config` (VitisAI config in `models/`), `--seq-len`, `--threads`.

Segments are `start end speaker_N`, with speakers numbered in order of first appearance; overlapping segments mean
overlapping speech. The model supports up to 8 speakers.

### NPU power mode

Turbo mode speeds up both this model and Parakeet substantially (offline diarization 109x to 175x real time):

```powershell
C:\Windows\System32\AMD\xrt-smi.exe configure --pmode turbo      # default, powersaver, balanced, performance, turbo
C:\Windows\System32\AMD\xrt-smi.exe examine -r platform          # shows "Power Mode"
```

## VitisAI configuration

Both configs use `optimize_level: 3`, `preferred_data_storage: auto` and the `ops-blocklist=SiLUBf16` flag from the
Parakeet setup (harmless here; this model has no SiLU).

- `vai_ep_config_accurate.json` (default) adds the front-end flags `use-kernels-accurate-mode=on
  use-accurate-mode=on` through `fe_experiment` (flags are space separated). This roughly halves the per-step error
  vs FP32 for about 3% speed.
- `vai_ep_config.json` is the plain config (about 3% faster, less accurate).

What did not help: `enable-bfp16-wts-gemm=off` (an `fe_experiment` flag) and `emulate_brain_fp_with_block_fp: false`
(a `vaiml_config` key) both gave bit-identical output. In SDK 1.8.0 the matmul kernels (`GemmBfp16`) appear to always
use block floating point for activations; a CPU emulation of that format (`diag_emul.py`) predicts about the
remaining gap to true BF16.

## Test data

- `audio/diarization_example.wav`: real 98 s, 6-speaker recording from the Transformers test data
  (`hf-internal-testing/dummy-audio-samples`, `diarization_example.mp3`, converted to WAV).
- `audio/synthetic_5spk*.wav`: 9 synthetic 5-speaker conversations with exact ground truth (`*.truth.json`),
  built by `make_test_audio.py [seed ...]` from LibriSpeech and other clips in `C:\Dev\RyzenAI-SW`,
  `C:\Dev\whisper.cpp` and `C:\Dev\WhisperKit` (turns with gaps and about 25% overlapping barge-ins).

## Scripts

| Script | Env | Purpose |
|---|---|---|
| `export_onnx.py` | export | Export a static-shape step graph (`--seq-len`, `--parts` to split it) |
| `diarizer.py` | ryzen-ai | Offline pipeline and CLI |
| `stream_diarizer.py` | ryzen-ai | Streaming pipeline (`StreamingDiarizer.feed()`) and file-as-stream CLI |
| `reference_hf.py`, `reference_stream_hf.py` | export | PyTorch references (offline, streaming) as `.ref.npz` |
| `reference_bf16.py`, `diag_bf16.py`, `diag_emul.py` | export | BF16 / block floating point error studies |
| `diag_step.py` | ryzen-ai | NPU vs CPU step error on identical inputs |
| `diag_mic.py` | ryzen-ai | Microphone device, host API and per-channel levels |
| `diag_repeats.py` | ryzen-ai | Transcribe a long file and count repeated phrases at chunk joints |
| `eval_der.py`, `eval_set.py`, `eval_stream.py` | either | DER scoring |
| `make_test_audio.py` | export | Synthetic conversations with ground truth |

## Known limitations

- The live app labels each transcript segment (split at pauses, at most 15 s) with its dominant speaker, plus a
  second one holding at least 20% of it; it does not split lines at word level. Its end-to-end test has only been
  run partially (the first line was transcribed and labelled correctly).
- Compiles take 35-60 minutes and fail if the process is interrupted. One compile also failed once on a transient
  `Cannot open layer control parameters binary file` error and succeeded on retry.
- The streaming labels deviate more from full precision on real audio (8% DER vs the streaming reference on the real
  clip) than the offline mode does.

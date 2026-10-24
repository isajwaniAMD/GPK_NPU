# nav_perception on the Ryzen AI NPU: run log, results, and optimisation options

Date: 2026-10-23
Machine: HP Z2 Mini G1a, AMD Ryzen AI MAX+ PRO 395 (Radeon 8060S iGPU, XDNA2 NPU at `/dev/accel0`), 32 CPU cores, 62 GB RAM
OS: Ubuntu 26.04.1, kernel 7.0.0-38, XRT 2.25.37
Software: Ryzen AI 1.8 venv (`~/ryzenai/venv`, Python 3.12), `VitisAIExecutionProvider`, activated with `source ~/ryzenai/env.sh`

## 1. Goal

Run `models/nav_perception.onnx` on the NPU, check it against the CPU, and measure latency.

## 2. Model and code in the directory

| File | Purpose |
|---|---|
| `models/nav_perception.onnx` | Perception network (8.3 MB, opset 17, 549 nodes) |
| `nav_torch.py` | PyTorch twin of the ONNX (weights loaded from the ONNX; fp32/fp16/bf16, optional fused attention) for the iGPU |
| `run_nav_model.py` | Runs the ONNX on CPU and NPU, compares outputs, times them, checks SDF geometry against an analytic sphere |

Model I/O: inputs `images[1,3,2,128,160]`, `valid_mask[1,1,128,160]`, `pose[1,4,4]`, `sdf_volume[1,1,32,32,32]`, `query_pts[1,256,3]`.
Outputs: `depth`, `conf`, `anchor_conf`, `anchor_idx`, `anchor_depth`, `sector_risk`, `scene_logits`, `sdf`, `sdf_grad`, `collision_cost`.

The "FFI robotic stack overview" document the user referred to was not available on this machine (its Windows path could not be reached from this Linux session), so none of it was read. `build_nav_model.py`, mentioned in `nav_torch.py`, is also not in the directory.

## 3. Steps taken, in order

1. **Operator check** (`python run_nav_model.py --check-only`): only `TopK` (1 node) is outside the BF16 operator table.
2. **Config:** copied `vai_ep_config.json` from `~/isdev/NPU_WL/parakeetnpu/models/` (VAIML target, `optimize_level 3`, `ops-blocklist=SiLUBf16`).
3. **First NPU run** (with a 580 s timeout, output piped through `tail`): the process ended with no NPU results. The script captures the compiler's native output in memory and writes it only after the session is built, so the error was lost.
4. **Second run, no timeout:** exited with code 255 and again no NPU output.
5. **Diagnosis:** built the session directly with ORT logging on and the output written to `session.log`. The real error appeared:
   ```
   ERROR: [aiecompiler 77-6416] par.subgraph.be.mlir:74851:5 : The size of buffer_dimension is 1515520,
   which should be less than or equal to SharedBuffer/ExternalBuffer(compute_graph.Layer_279_wts_ddr)
   dimensions which is 757760.
   Compilation Failed
   ```
6. **Mapping the failing layer:** `Layer_279` is `=Gemm->sector_risk`, the last op of the head
   `conv_706 -> Sigmoid(conf) -> Reshape[1,5120] -> Gemm(w[5120,7], b[7]) -> sector_risk`.
   A very tall and skinny matmul (K=5120, N=7). The compiler's own weight layout for it (1,515,520 elements) is exactly twice the DDR buffer it allocated (757,760), while the real weights are only 35,840 elements. This is a compiler-internal inconsistency, not a model error. Note: the partition summary reported `AIE 100% / CPU 0%`, but the backend compile still failed, so the summary is not proof of success.
7. **Fix chosen: run that Gemm on the CPU.**
   - `tools/split_sector_head.py` removes the Gemm from the ONNX and exposes its input as `sector_feat[1,5120]`. It writes `models/nav_perception_npu.onnx` and `models/sector_head.npz` (the Gemm weights).
   - `run_nav_model.py` `NavPerception` now detects `sector_feat` and computes `sector_risk = sector_feat @ w + b` in NumPy, so the returned dictionary is unchanged.
   - On the CPU the split model matches the original on every output; the only nonzero difference is `sector_risk` at 3.6e-7.
   - The original `nav_perception.onnx` is untouched.
8. **NPU run on the split model:**
   ```
   python run_nav_model.py --npu --model models/nav_perception_npu.onnx \
       --ref-model models/nav_perception.onnx --config-file $PWD/vai_ep_config.json --save out
   ```
   Compile succeeded in 441.6 s. Artifact: `npu_cache/nav_perception_npu/nav_perception_npu.rai`. Later runs load the cache in about 0.4 s.
9. **Placement:** the log reports `CPU 50`, `VAIML 497`, `NPU subgraphs 1`, `Actually running on NPU 1`. The script's "no operators on NPU" warning is a false alarm: its parser looks for the label `NPU` and the log says `VAIML`. This parser has not been fixed yet. NPU execution was inferred from the compiler log; live utilisation (`xrt-smi`) was not observed.
10. **Benchmark** (`tools/bench.py`, 300 timed runs after 10 warm-up calls, single frame, synthetic input).
11. **Profile** (ORT profiling on the NPU session, 30 runs) and per-layer compute breakdown from `npu_cache/nav_perception/gops.csv`.
12. **Deliverables:** `nav_perception_npu_vs_cpu.pptx` (built by `tools/make_pptx.py`; not opened, since no PowerPoint/LibreOffice is installed here).

## 4. Results

### 4.1 Latency, per frame

The model is a perception network (one forward pass per frame). It generates no tokens, so first-token and inter-token latency do not apply; first-call and steady-state per-frame latency are the equivalents.

| Metric | CPU (fp32) | NPU (BF16) |
|---|---|---|
| Session load, compile cached | 0.08 s | 0.39 s |
| First-call latency (cold) | 14.6 ms | 27.7 ms |
| Steady-state mean | 6.31 ms | 21.51 ms |
| p50 | 5.97 ms | 21.00 ms |
| p95 | 7.44 ms | 26.00 ms |
| p99 | 14.08 ms | 27.99 ms |
| Min / max | 5.85 / 17.40 ms | 19.87 / 30.04 ms |
| Throughput | ~159 frames/s | ~46 frames/s |
| One-time NPU compile | n/a | 442 s |

The earlier `run_nav_model.py` run gave NPU p50 24.0 / p95 27.8 / p99 36.1 ms and CPU p50 6.2 ms; run-to-run variation is a few ms.

### 4.2 Accuracy, NPU (BF16) vs CPU fp32

| Output | Cosine | Other |
|---|---|---|
| depth | 0.99961 | max error 1.9 m |
| conf | 0.99976 | |
| anchor_conf | 0.99997 | on anchors both runs picked |
| anchor_idx | n/a | 90.6% index overlap |
| anchor_depth | 0.99937 | max error 1.4 m |
| sector_risk | 0.99993 | |
| scene_logits | 0.99969 | |
| sdf | 0.99994 | mean error 0.011 m (CPU 0.008 m) |
| sdf_grad | 0.99986 | gradient cosine minimum 0.998 |
| collision_cost | 1.00000 | 28.50 vs 28.80 CPU vs 29.82 analytic |

Differences are in line with BF16. The planner-facing items to watch are the depth error (1.9 m) and the top-64 anchor overlap (90.6%).

### 4.3 Where the time goes

Profile of the NPU session, per frame:

| Part | Time |
|---|---|
| Single NPU subgraph (497 ops) | 17.5 ms |
| CPU fallback ops, all together (2 Conv, 3 Max, 8 Gather, 12 Add, AveragePool, TopK, ...) | about 2 ms |

About 90% of the time is the NPU executing the network. An earlier statement that CPU fallback and data movement were the main cause was wrong; fixing the fallbacks would gain at most about 2 ms.

Compute per frame is 3.2 GOPs (compiler `gops.csv`):

| Layer type | Share |
|---|---|
| Conv | 80% |
| ConvTranspose | 14% |
| MatMul / Gemm (attention blocks etc.) | about 5% |

Three convolutions (`conv_14` 0.76 GOPs, `conv_43` 0.59, `batchnormalization_27` 0.38) are about 54% of all compute, probably the high-resolution stem and early layers. 3.2 GOPs in 17.5 ms is about 0.18 TOPS, far below what the NPU can sustain, so the run is overhead-bound (layer scheduling, data movement, layouts), not arithmetic-bound.

## 5. Conclusions

- The model runs on the NPU after one change (the `sector_risk` Gemm on the CPU), with accuracy within BF16 expectations.
- The NPU is currently about 3.4x slower than the CPU for this model (21.5 ms vs 6.3 ms mean). The network is small and the CPU has 32 cores.
- The case for the NPU is power and freeing the CPU/GPU, which has not been measured.

## 6. How to optimise (quantization excluded, since this is a robotics model)

Items 3 to 6 change the network and would need re-validation or retraining. None of the options below has been tested.

1. **Compiler options (no model change).** Try lower `optimize_level`, the `vai_ep_config_accurate.json` from `~/isdev/NPU_WL/qwen3_asr/models/`, and removing the `SiLUBf16` blocklist. About 7 minutes per attempt. Likely a modest gain.
2. **Pipeline across frames (no model change).** Run CPU post-processing (top-k, gathers, SDF lookup, the `sector_risk` Gemm) for frame N while the NPU runs frame N+1. Throughput rises; single-frame latency does not.
3. **Lighten the stem.** The three heaviest layers are over half the compute: use a stride-2 stem, lower input resolution, or fewer early channels. Replace 5x5 convolutions with two stacked 3x3.
4. **Replace ConvTranspose** (14% of compute) with `Resize` + Conv, or `DepthToSpace` (in the supported table).
5. **Use channel counts that are multiples of 16 or 32** for better compiler tiling; may help the low utilisation.
6. **Use the iGPU.** `nav_torch.py` is the PyTorch twin for the Radeon 8060S (fp16/bf16, optional fused attention via `attn="sdpa"`, `sdf="grid"`). It avoids quantization and may beat both CPU and NPU on latency. This has not been benchmarked.
7. **Fix the `sector_risk` shape instead of moving it to the CPU** (padding N from 7 to 8/16/32 with a Slice, or splitting K into chunks). This would put the head back on the NPU but is likely worth under 1 ms.

The attention blocks are only about 5% of compute, so rewriting attention for the NPU is not a priority.

Suggested order: measure power on CPU, iGPU and NPU first (this decides whether the latency gap matters), then try the compiler options in item 1, then item 6 for a latency comparison, then the stem changes in item 3.

## 7. Not done / caveats

- No live NPU utilisation (`xrt-smi`) observed; placement is from the compiler log.
- No power or energy measurement; no batching test; the iGPU path was not run.
- Inputs are synthetic (random images, a sphere SDF), not real camera frames.
- The `run_nav_model.py` parser still mislabels `VAIML` operators as missing NPU operators.
- The compile log is only written after a successful session build; on a compiler error it is lost. Use the direct-session snippet (see step 5) to see errors.

## 8. Files created or changed

| Path | What |
|---|---|
| `vai_ep_config.json` | Copied from the parakeetnpu project |
| `tools/split_sector_head.py` | Cuts the `sector_risk` Gemm out of the ONNX |
| `models/nav_perception_npu.onnx`, `models/sector_head.npz` | NPU model and CPU head weights |
| `run_nav_model.py` | `NavPerception` applies the CPU head when `sector_feat` is an output |
| `tools/bench.py` | Latency benchmark (CPU and NPU) |
| `tools/make_pptx.py`, `nav_perception_npu_vs_cpu.pptx` | Management slides |
| `npu_cache/` | Compile caches (`nav_perception/` from the failed compile, `nav_perception_npu/` from the working one) |
| `out/nav_run.json`, `out/nav_io.npz` | Saved run results and I/O |
| `session.log`, `nav_npu_compile.log`, `run_npu*.log` | Logs from the runs above |

To reproduce: `source ~/ryzenai/env.sh`, then the command in step 8 of section 3.


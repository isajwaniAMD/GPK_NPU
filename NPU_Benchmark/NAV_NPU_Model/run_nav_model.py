#!/usr/bin/env python3
"""
Run nav_perception.onnx on the CPU and the Ryzen AI NPU, check it, and reuse it.

Run inside the Ryzen AI Python environment you already used for your first model
(Linux: activate the venv, then `source /opt/xilinx/xrt/setup.sh`).

  python run_nav_model.py --check-only                      # op coverage vs the BF16 table, no NPU
  python run_nav_model.py                                   # CPU reference + sanity checks
  python run_nav_model.py --npu --config-file <vai_ep_config.json>

What it reports:
  * how many operators the Vitis AI compiler placed on the NPU and how many fell back to the CPU
  * CPU vs NPU latency (p50 / p95 / p99)
  * NPU vs CPU agreement per output (cosine, relative error; anchor-index overlap)
  * a geometry check of the SDF / gradient / collision-cost outputs against the analytic sphere
    in the synthetic map, which verifies the pose transform and the lookup

Reuse in the OKVIS2 pipeline: import NavPerception and call it once per frame (see the class).
"""
import argparse
import contextlib
import json
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

S, EXTENT = 32, 4.0
BF16_OK = set("""Abs Add And ArgMax ArgMin AveragePool BatchNormalization Cast Ceil Concat Constant ConstantOfShape
Conv ConvTranspose DepthToSpace Div Einsum Equal Erf Exp Expand Flatten Floor Gather GatherElements Gemm
GlobalAveragePool Greater GridSample GroupConv Identity InstanceNormalization Less Log MatMul Max MaxPool Min Mod
Mul Neg Not Or Pad Pow Reciprocal ReduceMax ReduceMean ReduceMin ReduceSum Reshape Resize Round ScatterND
Shape Sigmoid Sign Sin Size Slice Split Sqrt Squeeze Sub Tanh Tile Transpose Unsqueeze Where Xor""".split())


# ------------------------------------------------------------------------------------------
class NavPerception:
    """Thin wrapper to reuse in the full pipeline.

    nav = NavPerception("models/nav_perception.onnx", use_npu=True, config_file="vai_ep_config.json")
    out = nav(images, valid_mask, pose, sdf_volume, query_pts)     # dict of numpy arrays
    """

    def __init__(self, path, use_npu=False, config_file=None, cache_dir="npu_cache", extra=None, log_level=1):
        so = ort.SessionOptions()
        so.log_severity_level = log_level if use_npu else 3     # CPU reference session stays quiet
        if use_npu:
            if "VitisAIExecutionProvider" not in ort.get_available_providers():
                sys.exit("VitisAIExecutionProvider not available: activate the Ryzen AI environment first")
            opts = {"cache_dir": str(Path(cache_dir).resolve()), "cache_key": Path(path).stem}
            if config_file:
                opts["config_file"] = config_file
            opts.update(extra or {})
            self.sess = ort.InferenceSession(str(path), so, providers=["VitisAIExecutionProvider"],
                                             provider_options=[opts])
        else:
            self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.in_names = [i.name for i in self.sess.get_inputs()]
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.head = None
        if "sector_feat" in self.out_names:       # split model: sector_risk Gemm runs here, on the CPU
            h = np.load(Path(path).with_name("sector_head.npz"))
            self.head = (h["w"], h["b"])

    def __call__(self, images, valid_mask, pose, sdf_volume, query_pts):
        feeds = dict(zip(self.in_names, [images, valid_mask, pose, sdf_volume, query_pts]))
        out = dict(zip(self.out_names, self.sess.run(None, feeds)))
        if self.head is not None:
            out["sector_risk"] = out.pop("sector_feat") @ self.head[0] + self.head[1]
        return out


@contextlib.contextmanager
def capture_native_output():
    """Capture C-level stdout/stderr (ORT and the Vitis AI compiler log there)."""
    tmp = tempfile.TemporaryFile(mode="w+b")
    sys.stdout.flush(); sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    os.dup2(tmp.fileno(), 1); os.dup2(tmp.fileno(), 2)
    box = {}
    try:
        yield box
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        for fd in saved:
            os.close(fd)
        tmp.seek(0)
        box["text"] = tmp.read().decode("utf-8", errors="replace")
        tmp.close()


def parse_vitis_stats(text):
    m = re.search(r"No\. of Operators :(.*?)(No\. of Subgraphs|$)", text, re.S)
    npu = cpu = None
    if m:
        for line in m.group(1).splitlines():
            mm = re.search(r"\b(NPU|VITIS_EP_CPU|CPU)\s+(\d+)\s*$", line.strip())
            if mm:
                if mm.group(1) == "NPU":
                    npu = int(mm.group(2))
                else:
                    cpu = (cpu or 0) + int(mm.group(2))
    return npu, cpu


# ------------------------------------------------------------------------------------------
SPHERE_C, SPHERE_R = np.array([2.0, 0.0, 0.0]), 1.0        # synthetic obstacle in the map frame


def synthetic_scene(seed=0):
    rng = np.random.default_rng(seed)
    images = rng.uniform(0, 255, (1, 3, 2, 128, 160)).astype(np.float32)
    valid = np.ones((1, 1, 128, 160), np.float32)
    valid[:, :, :24, :] = 0                                       # "sky" rows masked out by OKVIS2's CNN
    th = np.deg2rad(30)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
    pose[:3, 3] = [0.5, 0.0, 0.2]
    pose = pose[None]
    ax = np.linspace(-EXTENT, EXTENT, S, dtype=np.float32)       # voxel corners, z,y,x order
    zz, yy, xx = np.meshgrid(ax, ax, ax, indexing="ij")
    vol = (np.sqrt((xx - SPHERE_C[0]) ** 2 + (yy - SPHERE_C[1]) ** 2 + (zz - SPHERE_C[2]) ** 2) - SPHERE_R)
    vol = vol.astype(np.float32)[None, None]
    q = rng.uniform(-1.5, 1.5, (1, 256, 3)).astype(np.float32)
    q[0, :, 0] += 1.5                                             # bias the samples toward the obstacle
    return images, valid, pose, vol, q


def geometry_check(out, pose, q):
    p = (pose[0, :3, :3] @ q[0].T).T + pose[0, :3, 3]            # body -> map frame
    d = p - SPHERE_C
    r = np.linalg.norm(d, axis=1)
    sdf_ref = r - SPHERE_R
    grad_ref = d / r[:, None]
    cost_ref = np.maximum(0.4 - sdf_ref, 0).sum()
    inside = np.all(np.abs(p) < EXTENT - 0.3, axis=1)
    e_sdf = np.abs(out["sdf"][0] - sdf_ref)[inside]
    cosg = np.sum(out["sdf_grad"][0] * grad_ref, 1) / (np.linalg.norm(out["sdf_grad"][0], axis=1) + 1e-9)
    return {"sdf_mean_abs_err_m": float(e_sdf.mean()), "sdf_max_abs_err_m": float(e_sdf.max()),
            "grad_cosine_min": float(cosg[inside].min()), "cost_model": float(out["collision_cost"][0, 0]),
            "cost_analytic": float(cost_ref)}


def compare(ref, got):
    rows = []
    for k in ref:
        r, g = ref[k], got[k]
        if k in ("anchor_conf", "anchor_depth") and "anchor_idx" in ref:
            # Ties in reduced precision can reorder or swap anchors; compare only anchors both runs picked.
            ra = dict(zip(ref["anchor_idx"].ravel().tolist(), r.ravel().tolist()))
            ga = dict(zip(got["anchor_idx"].ravel().tolist(), g.ravel().tolist()))
            common = sorted(set(ra) & set(ga))
            r, g = np.array([ra[i] for i in common]), np.array([ga[i] for i in common])
            k = f"{k}*"
        if r.dtype.kind in "iub":
            sa, sb = set(r.ravel().tolist()), set(g.ravel().tolist())
            rows.append((k, f"index overlap {len(sa & sb) / max(1, len(sa)):.1%}"))
        else:
            a, b = r.astype(np.float64).ravel(), g.astype(np.float64).ravel()
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-12))
            rows.append((k, f"cosine {cos:.5f}  rel-L2 {rel:.4g}  max|err| {np.abs(a - b).max():.4g}"))
    return rows


def timed(nav, args, warmup, runs):
    for _ in range(warmup):
        out = nav(*args)
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); out = nav(*args); ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    pct = lambda p: ts[min(len(ts) - 1, int(round(p / 100 * (len(ts) - 1))))]
    return out, {"p50": pct(50), "p95": pct(95), "p99": pct(99), "mean": statistics.fmean(ts)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/nav_perception.onnx")
    ap.add_argument("--ref-model", default=None, help="FP32 model for the CPU reference, e.g. models/nav_perception.onnx "
                    "when --model is the Quark BF16 model or an EP-context model")
    ap.add_argument("--npu", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--config-file", default=None, help="vai_ep_config.json from your working setup")
    ap.add_argument("--vai-option", action="append", default=[], help="extra provider option KEY=VALUE")
    ap.add_argument("--cache-dir", default="npu_cache")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--log-level", type=int, default=0, help="ORT severity for the NPU session (0 shows op placement)")
    ap.add_argument("--save", default=None, help="write nav_run.json and nav_io.npz here")
    a = ap.parse_args()

    m = onnx.load(a.model)
    hist = {}
    for n in m.graph.node:
        hist[n.op_type] = hist.get(n.op_type, 0) + 1
    opset = [o.version for o in m.opset_import if o.domain in ("", "ai.onnx")][0]
    unlisted = {k: v for k, v in hist.items() if k not in BF16_OK}
    print(f"model {a.model}: opset {opset}, {sum(hist.values())} nodes")
    print("not in the BF16 operator table:", ", ".join(f"{k}x{v}" for k, v in unlisted.items()) or "none")
    if a.check_only:
        return

    scene = synthetic_scene()
    images, valid, pose, vol, q = scene
    result = {"model": a.model}

    cpu = NavPerception(a.ref_model or a.model, False)      # FP32 reference (use --ref-model for BF16 / context models)
    ref, t = timed(cpu, scene, a.warmup, a.runs)
    print(f"\nCPU  p50 {t['p50']:.1f} ms   p95 {t['p95']:.1f} ms")
    result["cpu_ms"] = t
    g = geometry_check(ref, pose, q)
    print("geometry (CPU, fp32): SDF mean err %.3f m (max %.3f), gradient cosine min %.3f, "
          "collision cost %.2f vs analytic %.2f" % (g["sdf_mean_abs_err_m"], g["sdf_max_abs_err_m"],
                                                      g["grad_cosine_min"], g["cost_model"], g["cost_analytic"]))
    d = ref["depth"]
    print(f"depth range {d.min():.2f}..{d.max():.2f} m   anchors: top-{ref['anchor_idx'].shape[1]} "
          f"confidence {ref['anchor_conf'].min():.2f}..{ref['anchor_conf'].max():.2f}")

    if a.npu:
        extra = dict(kv.split("=", 1) for kv in a.vai_option)
        t0 = time.perf_counter()
        with capture_native_output() as cap:
            npu = NavPerception(a.model, True, a.config_file, a.cache_dir, extra, a.log_level)
        build_s = time.perf_counter() - t0
        Path("nav_npu_compile.log").write_text(cap["text"], encoding="utf-8")
        n_npu, n_cpu = parse_vitis_stats(cap["text"])
        print(f"\nNPU session ready in {build_s:.1f} s (compile cached in {a.cache_dir}/); log: nav_npu_compile.log")
        print(f"session providers: {npu.sess.get_providers()}")
        print(f"operator placement: NPU {n_npu}  CPU fallback {n_cpu}   (None = stats not found, read the log)")
        if not n_npu:
            print("\n*** WARNING: no operators were reported on the NPU. This run is NOT evidence of NPU execution; "
                  "ORT silently falls back to the CPU for anything the Vitis AI EP does not take. ***\n"
                  "Check nav_npu_compile.log for the '[Vitis AI EP] No. of Operators' block and for compile errors, "
                  "and watch NPU utilisation (Task Manager > Performance > NPU on Windows, `xrt-smi` on Linux) during the run.\n")
        got, tn = timed(npu, scene, a.warmup, a.runs)
        print(f"NPU  p50 {tn['p50']:.1f} ms   p95 {tn['p95']:.1f} ms   p99 {tn['p99']:.1f} ms   "
              f"(CPU p50 {t['p50']:.1f} ms)")
        for k, s in compare(ref, got):
            print(f"  {k:15s} {s}")
        gn = geometry_check(got, pose, q)
        print("geometry (NPU): SDF mean err %.3f m (max %.3f), gradient cosine min %.3f, "
              "collision cost %.2f" % (gn["sdf_mean_abs_err_m"], gn["sdf_max_abs_err_m"], gn["grad_cosine_min"],
                                       gn["cost_model"]))
        result.update({"npu_ms": tn, "ops_npu": n_npu, "ops_cpu_fallback": n_cpu, "geometry_npu": gn})
    if a.save:
        Path(a.save).mkdir(parents=True, exist_ok=True)
        (Path(a.save) / "nav_run.json").write_text(json.dumps(result, indent=2))
        extra_out = {f"npu_{k}": v for k, v in got.items()} if a.npu else {}
        np.savez(Path(a.save) / "nav_io.npz", images=images, valid_mask=valid, pose=pose, sdf_volume=vol,
                 query_pts=q, **{f"cpu_{k}": v for k, v in ref.items()}, **extra_out)
        print("saved", a.save)


if __name__ == "__main__":
    main()

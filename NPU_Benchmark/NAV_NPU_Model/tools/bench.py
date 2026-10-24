import sys, time, statistics as st
sys.path.insert(0, ".")
import run_nav_model as r
sc = r.synthetic_scene()
def stats(ts):
    ts = sorted(ts); p = lambda q: ts[min(len(ts)-1, int(round(q/100*(len(ts)-1))))]
    return dict(first=None, mean=st.fmean(ts), p50=p(50), p95=p(95), p99=p(99), mn=ts[0], mx=ts[-1])
for name, model, npu in [("CPU", "models/nav_perception.onnx", False), ("NPU", "models/nav_perception_npu.onnx", True)]:
    t0 = time.perf_counter()
    nav = r.NavPerception(model, npu, "vai_ep_config.json" and __import__("os").path.abspath("vai_ep_config.json"), log_level=3)
    load = time.perf_counter() - t0
    t = time.perf_counter(); nav(*sc); first = (time.perf_counter() - t) * 1e3
    for _ in range(10): nav(*sc)
    ts = []
    for _ in range(300):
        t = time.perf_counter(); nav(*sc); ts.append((time.perf_counter() - t) * 1e3)
    s = stats(ts)
    print(f"{name} load_s={load:.2f} first_ms={first:.1f} mean={s['mean']:.2f} p50={s['p50']:.2f} p95={s['p95']:.2f} p99={s['p99']:.2f} min={s['mn']:.2f} max={s['mx']:.2f} fps={1000/s['mean']:.0f}")

"""Quick check that PyTorch (ROCm) really runs on the iGPU:  .venv-igpu/bin/python check_igpu.py"""
import time
from pathlib import Path

import torch

print("torch", torch.__version__, "| HIP", torch.version.hip, "| GPU available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("No GPU visible. Is your user in the 'render' group (id -nG) and did you log in again?")
print("device:", torch.cuda.get_device_name(0), "| arch:", torch.cuda.get_device_properties(0).gcnArchName)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
busy = Path("/sys/class/drm/card0/device/gpu_busy_percent")
peak = 0
for _ in range(60):
    for _ in range(20):
        y = x @ x
    torch.cuda.synchronize()
    peak = max(peak, int(busy.read_text()) if busy.exists() else -1)
print("fp16 matmul ok, checksum", float(y.float().mean()), "| gpu_busy peak", peak, "%")

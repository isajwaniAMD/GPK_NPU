#!/usr/bin/env bash
# LLM on the iGPU (PyTorch ROCm). Usage: ./start_llm_server.sh [local-folder]  (default: <repo root>/models/Qwen3-4B-Instruct-2507)
set -euo pipefail
cd "$(dirname "$0")"
unset PYTORCH_HIP_ALLOC_CONF            # backend:malloc is known to crash ROCm PyTorch
source .venv-igpu/bin/activate
exec python llm_server.py --model "${1:-$(cd .. && pwd)/models/Qwen3-4B-Instruct-2507}" --port 8008

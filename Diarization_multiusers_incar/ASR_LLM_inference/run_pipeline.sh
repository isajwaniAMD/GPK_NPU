#!/usr/bin/env bash
# ASR on the NPU, answers from the LLM server. Usage: ./run_pipeline.sh [wav]
set -euo pipefail
cd "$(dirname "$0")"
source ~/ryzenai/env.sh >/dev/null 2>&1
exec python asr_llm_pipeline.py "${1:-incar_commands_us_en.wav}"

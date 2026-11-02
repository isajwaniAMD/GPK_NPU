#!/usr/bin/env bash
# Start the LLM server on the iGPU (if not already up), then the browser demo. Usage: ./run_demo.sh [wav]
# Then open http://127.0.0.1:8080
cd "$(dirname "$0")"
LLM=http://127.0.0.1:8008

if ! curl -fs -m 2 "$LLM/health" >/dev/null; then
  echo "starting LLM server on the iGPU ..."
  nohup ../ASR_LLM_inference/start_llm_server.sh > llm_server.log 2>&1 &
  for _ in $(seq 60); do curl -fs -m 2 "$LLM/health" >/dev/null && break; sleep 2; done
  curl -fs -m 2 "$LLM/health" >/dev/null || { echo "LLM server did not come up, see demo_gui/llm_server.log"; exit 1; }
fi

# the Ryzen AI venv's activate script reads these; make sure they exist
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}" PYTHONPATH="${PYTHONPATH-}"
source ~/ryzenai/env.sh >/dev/null 2>&1
exec python app.py "$@"

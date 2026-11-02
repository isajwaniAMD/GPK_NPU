#!/usr/bin/env bash
# Fetch (or build) every model this repo needs. Safe to re-run: finished files are skipped, interrupted downloads resume.
#
#   ./fetch_models.sh [all|llm|asr|diar] [--check] [--prune] [--force]
#
#   all    everything below (default)
#   llm    Qwen3-4B-Instruct-2507, the iGPU language model        -> models/Qwen3-4B-Instruct-2507/        (~8 GB download)
#   asr    Parakeet TDT 0.6B v3: download the ONNX originals, convert them to static shapes, fuse Pad->Conv for the NPU
#                                                               -> models/encoder-model.fp32.static.npu.onnx(+.data),
#                                                                  models/decoder_joint-model.fp32.static.onnx
#                                                                  (~2.5 GB download, ~10 GB disk while building)
#   diar   Nemotron-3-Diarization: download the weights, export the NPU step graph
#                                                               -> diarization/models/diar_step_L704.onnx, diar_aux.npz
#                                                                  (~0.4 GB download; export needs its own conda env)
#   --check   only report which required files are present or missing (no network, no changes)
#   --prune   after the ASR build, delete the intermediate files the runtime never reads (~4.7 GB)
#   --force   rebuild/re-download even if the final files exist
#
# Environment overrides:
#   PYTHON             python for the ASR build (needs onnx; default: python, falling back to ~/ryzenai/env.sh if onnx is missing)
#   DIAR_PYTHON        python for the diarizer export (default: "conda run -n nemo-diar-export python" when that env exists)
#   QWEN_REV, NEMOTRON_REV, PARAKEET_REV   Hugging Face revision (branch, tag or commit) to pin; default main
#
# Not fetched, on purpose: models/vai_cache and diarization/models/vai_cache. The compiled NPU binaries are built on the
# first run (35-60 min for the diarizer, ~30 min for the encoder) and are specific to your driver and Ryzen AI version.
#
# Note: diarizer.py and export_onnx.py hard-code diarization/models/, so the diarizer files live there, not in models/.
set -eo pipefail   # no -u: the Ryzen AI venv's activate script reads variables that may be unset

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS="$REPO/models"
DIAR_MODELS="$REPO/diarization/models"

QWEN_REPO="Qwen/Qwen3-4B-Instruct-2507";                QWEN_REV="${QWEN_REV:-main}"
PARAKEET_REPO="istupakov/parakeet-tdt-0.6b-v3-onnx";    PARAKEET_REV="${PARAKEET_REV:-main}"
NEMOTRON_REPO="nvidia/Nemotron-3-Diarization";          NEMOTRON_REV="${NEMOTRON_REV:-main}"
DIAR_SEQ_LEN=704                                         # the offline graph the pipeline uses

what=all; CHECK=0; PRUNE=0; FORCE=0
for a in "$@"; do
  case "$a" in
    all|llm|asr|diar) what="$a" ;;
    --check) CHECK=1 ;; --prune) PRUNE=1 ;; --force) FORCE=1 ;;
    -h|--help) sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $a (try --help)" >&2; exit 2 ;;
  esac
done

hf_url() { echo "https://huggingface.co/$1/resolve/$2/$3"; }

# fetch URL DEST: skip when the local size matches the server's, otherwise download with resume.
# curl is used (not the Hugging Face python client) because it works behind SSL inspection.
fetch() {
  local url="$1" dest="$2" remote local_size
  mkdir -p "$(dirname "$dest")"
  remote="$(curl -sIL --retry 3 "$url" | awk 'tolower($1)=="content-length:"{v=$2} END{print v}' | tr -d '\r')"
  if [ -f "$dest" ] && [ "$FORCE" -eq 0 ]; then
    local_size="$(stat -c %s "$dest")"
    if [ -n "$remote" ] && [ "$local_size" = "$remote" ]; then echo "  [skip] $(basename "$dest")"; return 0; fi
  fi
  [ "$FORCE" -eq 1 ] && rm -f "$dest"
  echo "  [down] $(basename "$dest") ${remote:+($((remote / 1048576)) MB)}"
  curl -fL -C - --retry 3 -o "$dest" "$url"
}

need() { # need FILE...: print present/missing, return 1 if any missing
  local ok=0 f
  for f in "$@"; do
    if [ -e "$f" ]; then printf '  [ ok ] %s\n' "${f#$REPO/}"; else printf '  [MISS] %s\n' "${f#$REPO/}"; ok=1; fi
  done
  return $ok
}

LLM_FILES=(config.json generation_config.json merges.txt model.safetensors.index.json tokenizer.json tokenizer_config.json vocab.json
           model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors)
ASR_FINAL=("$MODELS/encoder-model.fp32.static.npu.onnx" "$MODELS/encoder-model.fp32.static.npu.onnx.data"
           "$MODELS/decoder_joint-model.fp32.static.onnx" "$MODELS/static_config.json" "$MODELS/config.json" "$MODELS/vocab.txt")
DIAR_FINAL=("$DIAR_MODELS/diar_step_L${DIAR_SEQ_LEN}.onnx" "$DIAR_MODELS/diar_aux.npz")

report() {
  echo "== required model files"
  local rc=0
  echo " llm:";  need "${LLM_FILES[@]/#/$MODELS/Qwen3-4B-Instruct-2507/}" || rc=1
  echo " asr:";  need "${ASR_FINAL[@]}" || rc=1
  echo " diar:"; need "${DIAR_FINAL[@]}" || rc=1
  echo " (compiled NPU caches are built on the first run and are not listed)"
  return $rc
}

if [ "$CHECK" -eq 1 ]; then report; exit $?; fi

# ---------------------------------------------------------------------------------------------------------- llm
fetch_llm() {
  echo "== llm: $QWEN_REPO@$QWEN_REV"
  local f
  for f in "${LLM_FILES[@]}"; do fetch "$(hf_url "$QWEN_REPO" "$QWEN_REV" "$f")" "$MODELS/Qwen3-4B-Instruct-2507/$f"; done
}

# ---------------------------------------------------------------------------------------------------------- asr
ensure_python_onnx() {
  PYTHON="${PYTHON:-python}"
  if ! "$PYTHON" -c "import onnx" 2>/dev/null && [ -f "$HOME/ryzenai/env.sh" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}" PYTHONPATH="${PYTHONPATH-}"
    # shellcheck disable=SC1091
    source "$HOME/ryzenai/env.sh" >/dev/null 2>&1 || true
    PYTHON=python
  fi
  "$PYTHON" -c "import onnx" 2>/dev/null || { echo "ERROR: '$PYTHON' has no 'onnx' package. Activate the Ryzen AI environment (source ~/ryzenai/env.sh) or set PYTHON=." >&2; return 1; }
}

fetch_asr() {
  echo "== asr: $PARAKEET_REPO@$PARAKEET_REV"
  if [ "$FORCE" -eq 0 ] && need "${ASR_FINAL[@]}" >/dev/null 2>&1; then echo "  final files already present, skipping (use --force to rebuild)"; return 0; fi
  ensure_python_onnx || return 1
  local f
  # config.json and vocab.txt are small and tracked in git; the rest are the dynamic-shape originals the build starts from.
  for f in config.json vocab.txt encoder-model.onnx encoder-model.onnx.data decoder_joint-model.onnx; do
    fetch "$(hf_url "$PARAKEET_REPO" "$PARAKEET_REV" "$f")" "$MODELS/$f"
  done
  # The tools read ./models, so run them from the repo root.
  echo "  [build] static shapes (15 s chunks) ..."
  (cd "$REPO" && "$PYTHON" -m inference.tools.convert_static --models-dir models --precision fp32)
  echo "  [build] fuse Pad->Conv for the NPU compiler (also points static_config.json at the .npu model) ..."
  (cd "$REPO" && "$PYTHON" -m inference.tools.fuse_pads_direct)
  if [ "$PRUNE" -eq 1 ]; then
    echo "  [prune] removing intermediates the runtime never reads"
    rm -f "$MODELS"/encoder-model.onnx "$MODELS"/encoder-model.onnx.data "$MODELS"/decoder_joint-model.onnx \
          "$MODELS"/encoder-model.fp32.static.onnx "$MODELS"/encoder-model.fp32.static.onnx.data
  fi
}

# ---------------------------------------------------------------------------------------------------------- diar
fetch_diar() {
  echo "== diar: $NEMOTRON_REPO@$NEMOTRON_REV"
  if [ "$FORCE" -eq 0 ] && need "${DIAR_FINAL[@]}" >/dev/null 2>&1; then echo "  final files already present, skipping (use --force to re-export)"; return 0; fi
  local f
  for f in config.json processor_config.json model.safetensors; do
    fetch "$(hf_url "$NEMOTRON_REPO" "$NEMOTRON_REV" "$f")" "$DIAR_MODELS/hf/$f"
  done
  if [ -z "${DIAR_PYTHON:-}" ] && command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | awk '{print $1}' | grep -qx nemo-diar-export; then
    DIAR_PYTHON="conda run --no-capture-output -n nemo-diar-export python"
  fi
  if [ -z "${DIAR_PYTHON:-}" ]; then
    cat >&2 <<EOF
  [todo] weights are downloaded, but the ONNX export needs its own environment (it must NOT go into the Ryzen AI one):
           conda create -y -n nemo-diar-export python=3.12
           conda activate nemo-diar-export
           python -m pip install -r diarization/requirements-export.txt
           python diarization/export_onnx.py --seq-len $DIAR_SEQ_LEN
         or set DIAR_PYTHON to a python that has those packages and re-run: ./fetch_models.sh diar
EOF
    return 1
  fi
  echo "  [build] exporting diar_step_L${DIAR_SEQ_LEN}.onnx + diar_aux.npz (several minutes) ..."
  (cd "$REPO" && $DIAR_PYTHON diarization/export_onnx.py --seq-len "$DIAR_SEQ_LEN")
}

# ---------------------------------------------------------------------------------------------------------- run
status=0
case "$what" in
  llm)  fetch_llm || status=1 ;;
  asr)  fetch_asr || status=1 ;;
  diar) fetch_diar || status=1 ;;
  all)  fetch_llm || status=1; fetch_asr || status=1; fetch_diar || status=1 ;;
esac
echo
report || status=1
[ "$status" -eq 0 ] && echo "all required model files are present" || echo "some steps failed or files are missing; see above" >&2
exit "$status"

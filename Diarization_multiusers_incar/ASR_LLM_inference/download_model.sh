#!/usr/bin/env bash
# Download Qwen3-4B-Instruct-2507 with curl (the HF python client fails behind SSL inspection on this network).
# Usage: ./download_model.sh        (about 8 GB)
# Files go only to <repo root>/models/Qwen3-4B-Instruct-2507/ ; then start the server with: ./start_llm_server.sh
set -euo pipefail
cd "$(dirname "$0")"
repo="Qwen/Qwen3-4B-Instruct-2507"
dest="$(cd .. && pwd)/models/${repo##*/}"; mkdir -p "$dest"
files=$(curl -fsS "https://huggingface.co/api/models/${repo}" | python3 -c "
import json, sys
for s in json.load(sys.stdin)['siblings']:
    n = s['rfilename']
    if not n.endswith(('.md', '.gitattributes')) and not n.startswith('.'): print(n)")
for f in $files; do
  mkdir -p "$dest/$(dirname "$f")"
  echo "== $f"; curl -fL -C - --retry 3 -o "$dest/$f" "https://huggingface.co/${repo}/resolve/main/${f}"
done
echo "done: $dest"; du -sh "$dest"

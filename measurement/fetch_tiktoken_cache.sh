#!/usr/bin/env bash
# Populate a local tiktoken cache so the notebook runs without network access.
#
# These BPE rank files are downloaded from OpenAI's public endpoint rather than
# vendored into the repo, to avoid redistributing third-party data files.
# tiktoken names each cached file by the sha1 of its source URL.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p tkcache

urls=(
  "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
  "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
)

for url in "${urls[@]}"; do
  name=$(printf '%s' "$url" | shasum -a 1 | cut -d' ' -f1)
  if [ -s "tkcache/$name" ]; then
    echo "cached  $(basename "$url")"
    continue
  fi
  echo "fetching $(basename "$url")"
  curl -sSL --max-time 300 --retry 5 --retry-all-errors -o "tkcache/$name" "$url"
done

echo "done. tkcache/ holds $(ls tkcache | wc -l | tr -d ' ') files"

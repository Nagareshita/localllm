#!/usr/bin/env bash
set -euo pipefail

echo "== Qwen Local API (server) =="

# config 探索: 明示指定がなければ server/config.json を使う
if [ -z "${QWEN_CFG_PATH:-}" ]; then
  if [ -f "server/config.json" ]; then
    export QWEN_CFG_PATH="$(pwd)/server/config.json"
  elif [ -f "config.json" ]; then
    export QWEN_CFG_PATH="$(pwd)/config.json"
  else
    echo "config.json が見つかりません。server/config.json を作成してください。"
    exit 1
  fi
fi

echo "CFG: $QWEN_CFG_PATH"
echo "Starting uvicorn on 0.0.0.0:8003"

# uvx があれば優先
if command -v uvx >/dev/null 2>&1; then
  uvx uvicorn server.api:app --host 0.0.0.0 --port 8003
else
  uvicorn server.api:app --host 0.0.0.0 --port 8003
fi

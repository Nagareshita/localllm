#!/usr/bin/env bash
set -euo pipefail

# 汎用 API サーバー起動スクリプト
# - スクリプト冒頭の「User Config」を編集すると既定値を変更できます
# - 環境変数での上書きも可能（CLI引数は最優先）
# - 未知の引数は api_server.py にそのままパススルー

# ---------------------
# User Config（ここを編集）
# ---------------------
# 既定値（空欄可）。環境変数が設定されている場合はそちらが優先されます。
: "${QUANTIZATION:=8bit}"                          # 8bit | 4bit | none
: "${PORT:=8003}"
: "${MODEL_ID:=Qwen/Qwen2.5-Coder-7B-Instruct}"
: "${HOST:=0.0.0.0}"
: "${MODEL_DIR:=./models}"
: "${GPU_MEM_LIMIT:=}"                              # 例: 11.5GiB / 空なら未指定
: "${CPU_MEM_LIMIT:=}"                              # 例: 48GiB / 空なら未指定
: "${OFFLOAD_DIR:=}"                                # 例: ./models/offload / 空なら未指定

# 以降は通常変更不要
QUANTIZATION="$QUANTIZATION"
PORT="$PORT"
MODEL_ID="$MODEL_ID"
HOST="$HOST"
MODEL_DIR="$MODEL_DIR"
GPU_MEM_LIMIT="$GPU_MEM_LIMIT"
CPU_MEM_LIMIT="$CPU_MEM_LIMIT"
OFFLOAD_DIR="$OFFLOAD_DIR"

if [[ $# -gt 0 ]]; then
  echo "[ERROR] このスクリプトは引数を受け付けません。start_server.sh の 'User Config' を編集してから実行してください." >&2
  exit 1
fi

echo "================================================"
echo "OpenAI-Compatible API Server"
echo "================================================"
echo "Model ID     : $MODEL_ID"
echo "Quantization : $QUANTIZATION"
echo "Port         : $PORT"
echo "Host         : $HOST"
echo "Model Dir    : $MODEL_DIR"
[[ -n "$GPU_MEM_LIMIT" ]] && echo "GPU Mem Limit: $GPU_MEM_LIMIT"
[[ -n "$CPU_MEM_LIMIT" ]] && echo "CPU Mem Limit: $CPU_MEM_LIMIT"
[[ -n "$OFFLOAD_DIR"  ]] && echo "Offload Dir  : $OFFLOAD_DIR"
echo "================================================"
echo ""

CMD=(
  python api_server.py
  --quantization "$QUANTIZATION"
  --port "$PORT"
  --model-dir "$MODEL_DIR"
  --model "$MODEL_ID"
  --host "$HOST"
)

[[ -n "$GPU_MEM_LIMIT" ]] && CMD+=( --gpu-mem-limit "$GPU_MEM_LIMIT" )
[[ -n "$CPU_MEM_LIMIT" ]] && CMD+=( --cpu-mem-limit "$CPU_MEM_LIMIT" )
[[ -n "$OFFLOAD_DIR"  ]] && CMD+=( --offload-dir "$OFFLOAD_DIR" )

"${CMD[@]}"

echo ""
echo "Server stopped."

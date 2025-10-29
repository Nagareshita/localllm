#!/bin/bash
# start.sh - Qwen APIサーバー簡単起動スクリプト

echo "🚀 Qwen All-in-One 起動スクリプト"
echo "=================================="

# 設定ファイル確認
if [ ! -f "config.json" ]; then
    echo "❌ config.jsonが見つかりません"
    echo "   以下の内容でconfig.jsonを作成してください:"
    echo ""
    cat << 'EOF'
{
  "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "torch_dtype": "bfloat16",
  "device_map": "auto",
  "quantization": "bnb-8bit",
  "use_chat_template": true,
  "system": "You are a helpful coding assistant. Answer in Japanese unless code is required.",
  "max_new_tokens": 512,
  "min_new_tokens": 0,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 0,
  "repetition_penalty": 1.05,
  "num_beams": 1,
  "do_sample": true,
  "no_repeat_ngram_size": 0,
  "length_penalty": 1.0,
  "early_stopping": false,
  "stop_words": []
}
EOF
    exit 1
fi

# Python確認
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3が見つかりません"
    exit 1
fi

# 必要なパッケージチェック
echo "📦 依存パッケージをチェック中..."
python3 -c "import torch, transformers, fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  必要なパッケージが不足しています"
    echo "   以下を実行してください:"
    echo "   pip install torch transformers fastapi uvicorn bitsandbytes"
    read -p "今すぐインストールしますか？ (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install torch transformers accelerate bitsandbytes fastapi uvicorn huggingface-hub
    else
        exit 1
    fi
fi

# 環境変数設定
export QWEN_CFG_PATH="config.json"

# 起動
echo ""
echo "✅ 起動準備完了"
echo "📍 http://localhost:8000 でWeb UIが開きます"
echo "📍 http://localhost:8000/health でヘルスチェック"
echo ""
echo "🔥 起動中..."
echo ""

uvicorn qwen_app_all_in_one:app --host 0.0.0.0 --port 8000

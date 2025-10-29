# start.ps1 - Qwen APIサーバー簡単起動スクリプト (Windows PowerShell)

Write-Host "🚀 Qwen All-in-One 起動スクリプト" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""

# 設定ファイル確認
if (-not (Test-Path "config.json")) {
    Write-Host "❌ config.jsonが見つかりません" -ForegroundColor Red
    Write-Host "   以下の内容でconfig.jsonを作成してください:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host @"
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
"@ -ForegroundColor Gray
    exit 1
}

# Python確認
try {
    python --version | Out-Null
} catch {
    Write-Host "❌ Pythonが見つかりません" -ForegroundColor Red
    exit 1
}

# 必要なパッケージチェック
Write-Host "📦 依存パッケージをチェック中..." -ForegroundColor Yellow
$checkPackages = python -c "import torch, transformers, fastapi" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  必要なパッケージが不足しています" -ForegroundColor Yellow
    Write-Host "   以下を実行してください:" -ForegroundColor Yellow
    Write-Host "   pip install torch transformers fastapi uvicorn" -ForegroundColor Gray
    
    $response = Read-Host "今すぐインストールしますか？ (y/n)"
    if ($response -eq "y" -or $response -eq "Y") {
        pip install torch transformers accelerate fastapi uvicorn huggingface-hub
    } else {
        exit 1
    }
}

# 環境変数設定
$env:QWEN_CFG_PATH = "config.json"

# 起動
Write-Host ""
Write-Host "✅ 起動準備完了" -ForegroundColor Green
Write-Host "📍 http://localhost:8000 でWeb UIが開きます" -ForegroundColor Cyan
Write-Host "📍 http://localhost:8000/health でヘルスチェック" -ForegroundColor Cyan
Write-Host ""
Write-Host "🔥 起動中..." -ForegroundColor Yellow
Write-Host ""

uvicorn qwen_app_all_in_one:app --host 127.0.0.1 --port 8000

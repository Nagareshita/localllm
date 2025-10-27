🧠 QwenApp 使い方ガイド（Windows/Conda 推奨）

Qwen2.5-Coder-7B-Instruct を完全ローカルで動かす API/CLI 両対応アプリです。はじめにモデルを取得した後は、ネットワーク遮断でもコード生成・会話が可能です。

📁 構成
qwen_app.py           ← メインスクリプト（API と CLI）
config.json           ← 設定（編集OK）
models/               ← モデル格納先（自動作成・ローカル運用）

⚙️ 1. セットアップ（Conda 環境）

PowerShell で以下を実行（Python 3.10 固定、CUDA 12.1 例）：

1) 仮想環境作成と有効化
  mamba create -n rag python=3.10 -y
  conda activate rag

2) PyTorch（CUDA 12.1 ビルド）
  mamba install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

3) pip を更新
  python -m pip install --upgrade pip

4) 依存ライブラリをインストール
  pip install -r requirements.txt

重要: Conda 環境では Windows の `py` ランチャーは使わず、常に `python` を使ってください。`py` は別の Python を起動し、`torch` が見つからない原因になります。

補足（pip のみで揃える場合の参考）
- CPU のみ: `pip install torch==2.3.1`
- CUDA 12.1: `pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1`
（本ガイドは Conda + CUDA 推奨）

🧰 2. uv/uvx の導入（任意）

PowerShell:
- 公式スクリプト: `powershell -ExecutionPolicy Bypass -Command "iwr -useb https://astral.sh/uv/install.ps1 | iex"`
- または Winget: `winget install --id AstralSoftware.UV -e`
確認: `uv --version` と `uvx --version`

🧩 3. 設定ファイル（config.json）
{
  "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "torch_dtype": "bfloat16",
  "device_map": "auto",

  "use_chat_template": true,
  "system": "You are a highly capable coding assistant. Answer in Japanese unless code is required.",

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

  "stop_words": [],
  "return_json": true
}

ヒント
- VRAM が厳しい場合は `torch_dtype` を `float16` に、あるいはより小さいモデルに変更。
- `device_map` は通常 `auto` でOK（単一GPUに自動割当）。
- チャット形式でのプロンプト整形は `use_chat_template` を有効に。

📦 4. モデルの取得と保存先

- 初回起動時に Hugging Face から自動ダウンロードし、`./models/<モデル名>/` に保存。
- `.safetensors` が存在すれば以降はローカルのみを使用。
- すでにダウンロード済みモデルを手動で配置しても利用可能です。

🔒 5. オフライン運用

一度モデルを取得したら、以降は通信を遮断できます（安全運用）。
- 環境変数を設定（PowerShell）: `setx TRANSFORMERS_OFFLINE 1`
- 確認: `echo $env:TRANSFORMERS_OFFLINE`（1 が表示されればOK）

💬 6. CLI（対話/1回生成）

対話モード
- `conda activate rag`
- `python qwen_app.py --config config.json`

1回だけ生成
- `python qwen_app.py --config config.json --once "FizzBuzzを書いて"`

例（出力イメージ）
Qwen Local Chat (Ctrl+C で終了)  / オフライン: OFF
> PythonでFizzBuzzを書いて
...

🌐 7. API サーバー

- 設定ファイルの場所を環境変数で指定（PowerShell）:
  `$env:QWEN_CFG_PATH = "$PWD\config.json"`
- 起動: `uvicorn qwen_app:app --host 127.0.0.1 --port 8000`

テスト（PowerShell 例）
`curl -s http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{"prompt":"PythonでFizzBuzz"}'`

レスポンス例（要約）
{
  "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "result": "...",
  "usage": {"prompt_tokens": 30, "completion_tokens": 96, "total_tokens": 126}
}

🧱 8. ディレクトリ構成（例）
📦 Local_LLM/
 ┣ qwen_app.py
 ┣ config.json
 ┗ models/
    ┗ Qwen2.5-Coder-7B-Instruct/
       ┣ config.json
       ┣ model-00001-of-00004.safetensors
       ┗ tokenizer.json

⚠️ 9. 注意事項
- 初回ダウンロードはネット接続が必要（以後は不要）。
- 7B モデルの bfloat16/float16 推論は目安として 14〜16GB VRAM を想定。
- Windows では `py` ではなく、必ず Conda 環境の `python` を使用。
- オフライン時は `TRANSFORMERS_OFFLINE=1` を設定してから起動。

🔧 10. トラブルシューティング
- Torch が見つからない: `conda activate rag` 後に `python` を使用。`python -c "import torch; print(torch.__version__)"` で確認。
- CUDA が使われない/初期化エラー: CUDA 対応の torch を入れているか確認（Conda の `pytorch-cuda` が推奨）。GPU ドライバも最新に。
- モデルが見つからない: `models/<モデル名>/` に `.safetensors` があるか確認。オフラインなら一度オンラインでDL。
- FastAPI が無い: `pip install fastapi uvicorn`。
- オフライン動作しない: PowerShell を再起動して環境変数を再読込。

✅ まとめ
- 完全ローカル動作: 可能（初回のみDL）
- 設定による制御: `config.json`
- API / CLI 両対応: 同一コード
- オフライン運用: `TRANSFORMERS_OFFLINE=1`
- Windows/Conda 対応: `Pathlib` 利用・手順整備

💡 拡張アイデア
- ストリーミング応答（SSE / WebSocket）
- vLLM 等のエンジン統合（高速化）
- 生成ログ保存（監査/再現性向上）

次のステップ
1) `config.json` を調整
2) 一度オンラインでモデルをDL
3) 必要なら `TRANSFORMERS_OFFLINE=1` 設定
4) `python qwen_app.py` でチャット開始

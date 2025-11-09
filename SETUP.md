# StarCoder2-15B Instruct API Setup Guide

このガイドでは、StarCoder2-15B Instruct版をOpenAI互換APIとして起動し、Qdrant履歴管理付きWebチャットアプリを使用する方法を説明します。
本リポジトリのAPIサーバーは Transformers + bitsandbytes を用いた実装に更新されています（OpenAI互換のエンドポイントを提供）。

---

## システム要件

- **GPU**: NVIDIA GPU (VRAM 24GB以上推奨、8bit量子化で約15GB使用)
- **OS**: Ubuntu 22.04 / WSL2
- **Python**: 3.12 (推奨)
- **CUDA**: 12.4以上 (12.6推奨)

---

## 1) 環境作成（Python 3.12）

Mambaで環境を作成します。

```bash
mamba create -n llmapi python=3.12 -y
conda activate llmapi
```
---

## 2) pip / uv の準備（推奨）

高速インストールのため `uv` を使用します。

```bash
python -m pip install --upgrade pip
python -m pip install uv

## 3) PyTorch + CUDA をインストール

### pip を使用する場合（推奨）:

```bash
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```

### PyTorchのインストール確認

インストール後、以下のコマンドでCUDAが正しく認識されているか確認してください：

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

期待される出力例：
```
PyTorch version: 2.6.0+cu126
CUDA available: True
CUDA version: 12.6
```

---

## 4) 依存パッケージのインストール

プロジェクトディレクトリに移動し、requirements.txtからインストールします。

```bash
cd localllm
uv pip install -r requirements.txt
```
### インストールされる主要パッケージ（抜粋）

- Transformers / Accelerate: モデル読み込みと `device_map="auto"`
- bitsandbytes: 8bit/4bit 量子化
- FastAPI / Uvicorn / Pydantic: OpenAI互換APIの提供
- PyTorch: CUDA対応ディープラーニングフレームワーク

---

## 5) モデルのダウンロード（自動）

初回起動時、Hugging Faceから自動でモデルがダウンロードされます。
ダウンロード先は `./models` ディレクトリです（15Bクラスで数十GB）。

すでにモデルをお持ちの場合は、`./models/hub/` 以下に配置してください。

---

## 起動方法

### A) API サーバーの起動（WSL）

#### 8bit量子化（デフォルト、推奨）

このスクリプトは「引数を受け付けません」。起動前に start_server.sh 冒頭の「User Config」を編集して設定してください。

1) start_server.sh の User Config を編集

```bash
# 例: Qwen 7B の 8bit, ポート8003, 12GB級GPU向けメモリ指定
QUANTIZATION=8bit
PORT=8003
MODEL_ID=Qwen/Qwen2.5-Coder-7B-Instruct
HOST=0.0.0.0
MODEL_DIR=./models
GPU_MEM_LIMIT=11.5GiB
CPU_MEM_LIMIT=48GiB
OFFLOAD_DIR=./models/offload
```

2) 実行（引数なし）

```bash
chmod +x ./start_server.sh
./start_server.sh
```

12GB級GPU（例: RTX 4070 Ti 12GB / RTX 2000 Ada 12GB）向けの推奨設定

- 8bit維持でVRAM不足を避けるため、CPUオフロードを使います（User Config の GPU/CPU メモリ上限と OFFLOAD_DIR を設定）。

#### 量子化なし（FP16）

User Config の `QUANTIZATION=none` にしてから、`./start_server.sh` を実行。

#### 4bit量子化（低VRAMの場合）

User Config の `QUANTIZATION=4bit` にしてから、`./start_server.sh` を実行。

または、Pythonスクリプトを直接実行：

```bash
python api_server.py --quantization 8bit --port 8003
```

**起動確認**: `http://localhost:8003/health` にアクセスし `{ "status": "ok" }` が表示されればOK。
OpenAI互換のエンドポイント例： `POST /v1/chat/completions`, `POST /v1/completions`（SSEストリーミング対応）。

---

### B) 履歴管理サーバーの起動（別ターミナル）

```bash
python history_server.py
```

デフォルトでポート8004で起動します。履歴データは `./qdrant_data` に保存されます。

**起動確認**: `http://localhost:8004/health` にアクセス。

---

### C) Webチャットアプリの起動（Windows）

1. `chat.html` をWindowsにコピー
2. ブラウザで `chat.html` を開く（ダブルクリック）
3. 設定欄でAPI URLとHistory API URLを確認
   - API URL: `http://localhost:8003/v1/chat/completions`
   - History API: `http://localhost:8004`

---

## 使い方

### チャット機能

1. メッセージを入力して「送信」ボタンをクリック
2. StarCoder2が応答を生成します
3. 画面上部の設定欄で API URL/Session ID に加え、`temperature` / `top_p` / `max_tokens` / `stop` / `stream` をGUI操作で変更できます（入力値は OpenAI互換リクエストにそのまま渡されます）。

### 履歴管理機能

- **履歴保存**: 現在の会話をQdrantに保存
- **履歴読込**: Session IDに紐づく過去の会話を読み込み
- **履歴削除**: Qdrantから履歴を完全削除
- **チャットクリア**: 現在の画面上の会話のみクリア（Qdrantは保持）

Session IDを変更することで、複数の会話を管理できます。

---

## 生成パラメータ（デフォルトと変更方法）

このAPIは OpenAI互換のボディでパラメータを指定できます。未指定の場合は以下のデフォルトが使われます。

- 共通デフォルト
  - `temperature`: 0.7
  - `top_p`: 0.95
  - `max_tokens`: 512（新規生成トークン数の上限）
  - `stream`: false（SSEストリーミング無効）
  - `stop`: なし（停止語なし）

### エンドポイント別の指定方法

- `POST /v1/chat/completions`
  - リクエストボディ例（非ストリーミング）
    ```json
    {
      "model": "bigcode/starcoder2-15b-instruct-v0.1",
      "messages": [
        {"role": "system", "content": "あなたは有能なAIアシスタントです。"},
        {"role": "user", "content": "FizzBuzzをPythonで書いて"}
      ],
      "temperature": 0.2,
      "top_p": 0.9,
      "max_tokens": 256,
      "stop": ["\n\n"],
      "stream": false
    }
    ```
  - ストリーミング（SSE）を有効化するには `"stream": true` を指定します。

- `POST /v1/completions`
  - リクエストボディ例（非ストリーミング）
    ```json
    {
      "model": "bigcode/starcoder2-15b-instruct-v0.1",
      "prompt": "Write a haiku about the sea",
      "temperature": 0.7,
      "top_p": 0.95,
      "max_tokens": 64,
      "stream": false
    }
    ```

### cURL例

- chat/completions（非ストリーミング）
  ```bash
  curl -s http://localhost:8003/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
          "model":"bigcode/starcoder2-15b-instruct-v0.1",
          "messages":[
            {"role":"user","content":"こんにちは。自己紹介して"}
          ],
          "temperature":0.5,
          "top_p":0.9,
          "max_tokens":128
        }'
  ```

- chat/completions（ストリーミング）
  ```bash
  curl -N http://localhost:8003/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
          "model":"bigcode/starcoder2-15b-instruct-v0.1",
          "messages":[{"role":"user","content":"短いジョークを言って"}],
          "stream":true
        }'
  ```

- completions（非ストリーミング）
  ```bash
  curl -s http://localhost:8003/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
          "model":"bigcode/starcoder2-15b-instruct-v0.1",
          "prompt":"List 3 colors:",
          "temperature":0.2,
          "max_tokens":32
        }'
  ```

### 注意事項

- `model` はサーバー起動時にロードしたモデルIDと一致している必要があります（一致しない場合は400エラー）。
- `max_tokens` は「新規生成トークン数」の上限です。長いプロンプト + 大きな `max_tokens` はVRAM使用量を増やします。
- `stop` で指定した文字列が出現した時点で出力を打ち切ります（複数指定可）。

---

## トラブルシューティング

### CUDA out of memory エラー

- 8bit量子化を使用してください: `./start_server.sh 8bit 8003`
- それでもエラーが出る場合は4bit量子化: `./start_server.sh 4bit 8003`

### WindowsからWSLのAPIに接続できない

WSLのIPアドレスを確認してください：

```bash
ip addr show eth0 | grep inet
```

ブラウザのAPI URL設定を `http://[WSL_IP]:8003/v1/chat/completions` に変更。

### モデルのダウンロードが遅い

初回起動時は約30GBのモデルをダウンロードするため時間がかかります。
Hugging Faceのミラーを使用する場合：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Qdrant接続エラー

`history_server.py` が起動していることを確認してください。
ポート8004が使用中の場合は、別のポートを指定：

```bash
# history_server.pyの最終行を編集
uvicorn.run(app, host="0.0.0.0", port=8005)
```

---

## ディレクトリ構成

```
starcoder2-project/
├── api_server.py          # APIサーバー本体
├── start_server.sh        # 起動スクリプト
├── history_server.py      # Qdrant履歴管理サーバー
├── chat.html              # Webチャットアプリ
├── requirements.txt       # Python依存パッケージ
├── models/                # モデルキャッシュ（自動生成）
└── qdrant_data/           # 履歴データ（自動生成）
```

---

## 参考情報

- **Transformers**: https://huggingface.co/docs/transformers
- **bitsandbytes**: https://github.com/TimDettmers/bitsandbytes
- **StarCoder2モデル**: https://huggingface.co/bigcode/starcoder2-15b-instruct-v0.1
- **Qdrant公式ドキュメント**: https://qdrant.tech/documentation/

---

## ライセンス

StarCoder2は商用利用可能なBigCode OpenRAIL-Mライセンスです。

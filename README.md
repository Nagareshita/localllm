# 🚀 Qwen All-in-One - 完全統合版

## 📦 必要なファイル

```
your-project/
├── qwen_app_all_in_one.py  # ⬅️ メインファイル（これ1つだけ！）
└── config.json             # ⬅️ 設定ファイル
```

---

## ⚙️ config.json

```json
{
  "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "torch_dtype": "bfloat16",
  "device_map": "auto",
  "quantization": "bnb-8bit",
  
  "system": "You are a helpful coding assistant. Answer in Japanese unless code is required.",
  
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 0,
  "repetition_penalty": 1.05,
  "do_sample": true,
  
  "stop_words": []
}
```

---

## 🔧 セットアップ

### WSL2の場合

```bash
# 依存パッケージインストール
pip install torch transformers accelerate bitsandbytes
pip install fastapi uvicorn huggingface-hub

# LlamaIndexも使う場合（オプション）
pip install llama-index llama-index-llms-openai-like llama-index-embeddings-huggingface sentence-transformers
```

---

## 🎯 使い方

### 1️⃣ CLIモード（対話）

```bash
python qwen_app_all_in_one.py --config config.json
```

出力例:
```
Qwen CLI Chat (Ctrl+C で終了)
> こんにちは
こんにちは！何かお手伝いできることはありますか？
> Pythonでフィボナッチ数列を書いて
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

### 2️⃣ 1回だけ生成

```bash
python qwen_app_all_in_one.py --config config.json --once "Pythonでhello world"
```

### 3️⃣ APIサーバー起動（WSL2推奨）

```bash
# 起動
uvicorn qwen_app_all_in_one:app --host 0.0.0.0 --port 8000

# 出力:
# ✅ モデル: Qwen/Qwen2.5-Coder-7B-Instruct
# ✅ デバイス: cuda:0
# INFO: Uvicorn running on http://0.0.0.0:8000
```

**Windowsブラウザから以下にアクセス:**

- **Web UI**: http://localhost:8000
- **API健全性チェック**: http://localhost:8000/health

### 4️⃣ Web UIでチャット

1. WSL2でAPIサーバーを起動:
   ```bash
   uvicorn qwen_app_all_in_one:app --host 0.0.0.0 --port 8000
   ```

2. Windowsのブラウザで開く:
   ```
   http://localhost:8000
   ```

3. チャット開始！🎉

![Web UI Preview](https://via.placeholder.com/800x400?text=Beautiful+Chat+Interface)

### 5️⃣ LlamaIndexテスト

```bash
# APIサーバーを起動しておく（別ターミナル）
uvicorn qwen_app_all_in_one:app --host 0.0.0.0 --port 8000

# 別ターミナルでテスト実行
python qwen_app_all_in_one.py --test-llamaindex
```

出力例:
```
🧪 LlamaIndexテストを開始...
⚙️  LlamaIndex設定中...
🔨 インデックス構築中...
🔍 クエリ実行中...

✅ LlamaIndexテスト成功!
質問: What is a resistor?
回答: A resistor is a basic electrical component with resistance R.

💡 この方法でModelicaのGRAGを構築できます！
```

---

## 🔗 LlamaIndexから使う（プログラマティック）

APIサーバーが起動していれば、別のPythonスクリプトから利用可能:

```python
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ローカルQwenに接続
Settings.llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key="dummy",
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    is_chat_model=True,
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Modelicaドキュメントでインデックス構築
documents = [...]  # あなたのModelicaファイル
index = VectorStoreIndex.from_documents(documents)

# クエリ実行
query_engine = index.as_query_engine()
response = query_engine.query("Resistorの実装例は？")
print(response.response)
```

---

## 🌐 APIエンドポイント一覧

| エンドポイント | 用途 | 例 |
|---------------|------|-----|
| `GET /` | Web UI | ブラウザで開く |
| `GET /health` | ヘルスチェック | `curl http://localhost:8000/health` |
| `POST /chat` | シンプルチャット | Web UIから使用 |
| `POST /v1/chat/completions` | OpenAI互換API | LlamaIndexから使用 |

### OpenAI互換API例

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

---

## 🎨 Web UIの機能

- ✅ リアルタイム対話
- ✅ 会話履歴保持（最大10ターン）
- ✅ トークン使用量表示
- ✅ レスポンシブデザイン
- ✅ 美しいグラデーション UI

---

## 💾 モデルのダウンロード

初回実行時に自動ダウンロードされます（約13GB）:

```
./models/
└── Qwen2.5-Coder-7B-Instruct/
    ├── model-00001-of-00004.safetensors
    ├── model-00002-of-00004.safetensors
    ├── model-00003-of-00004.safetensors
    ├── model-00004-of-00004.safetensors
    ├── config.json
    ├── tokenizer.json
    └── ...
```

**オフラインで使う場合:**
```bash
export TRANSFORMERS_OFFLINE=1
uvicorn qwen_app_all_in_one:app --host 0.0.0.0 --port 8000
```

---

## 🚀 パフォーマンス

| 設定 | VRAM | 推論速度 |
|------|------|---------|
| 8bit量子化 | ~8GB | 3-5秒/100トークン |
| 4bit量子化 | ~5GB | 4-6秒/100トークン |
| FP16 | ~14GB | 2-3秒/100トークン |

---

## 🐛 トラブルシューティング

### Q1: Windowsから接続できない

```bash
# WSL2のIPアドレス確認
ip addr show eth0

# Windowsから確認
curl http://<WSL2のIP>:8000/health
```

### Q2: メモリ不足

config.jsonで量子化を変更:
```json
{
  "quantization": "bnb-4bit"  // 8bit → 4bit
}
```

### Q3: bitsandbytesエラー

```bash
# WSL2
pip install bitsandbytes

# Windows（実験的）
pip install bitsandbytes-windows
```

---

## 📚 次のステップ

1. ✅ Web UIで動作確認
2. ✅ LlamaIndexテスト実行
3. ✅ 実際のModelicaファイルでGRAG構築
4. ✅ コード生成 + 検証ループの実装

---

## 🎉 すべてが1つのファイルに！

- ✅ CLI対話モード
- ✅ OpenAI互換API
- ✅ 美しいWeb UI
- ✅ LlamaIndex統合例
- ✅ 設定ファイル1つだけ

**たった2つのファイルで完全なローカルLLMシステムが動きます！** 🔥

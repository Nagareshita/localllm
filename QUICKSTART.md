# ⚡ 5分で始めるクイックスタート

## 📥 ステップ1: ファイルをダウンロード

以下の2ファイルだけあればOK:

```
your-project/
├── qwen_app_all_in_one.py
└── config.json
```

---

## ⚙️ ステップ2: config.json を作成

```json
{
  "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "torch_dtype": "bfloat16",
  "device_map": "auto",
  "quantization": "bnb-8bit",
  "system": "You are a helpful coding assistant.",
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "repetition_penalty": 1.05,
  "do_sample": true
}
```

---

## 📦 ステップ3: パッケージインストール

### WSL2 / Linux:

```bash
pip install torch transformers accelerate bitsandbytes
pip install fastapi uvicorn huggingface-hub
```

### Windows (CPU版):

```powershell
pip install torch transformers accelerate
pip install fastapi uvicorn huggingface-hub
```

---

## 🚀 ステップ4: 起動

### 方法A: 簡単起動スクリプト

**WSL2/Linux:**
```bash
chmod +x start.sh
./start.sh
```

**Windows PowerShell:**
```powershell
.\start.ps1
```

### 方法B: 直接起動

```bash
uvicorn qwen_app_all_in_one:app --host 0.0.0.0 --port 8000
```

---

## 🎉 ステップ5: 使ってみる

### Web UI（最も簡単！）

1. ブラウザで開く: **http://localhost:8000**
2. チャット開始！

### CLI対話モード

```bash
python qwen_app_all_in_one.py --config config.json
```

```
> こんにちは
こんにちは！何かお手伝いできることはありますか？

> Pythonでフィボナッチ数列
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

### LlamaIndexテスト

```bash
# 別ターミナルでAPIサーバーを起動しておく
uvicorn qwen_app_all_in_one:app --host 0.0.0.0 --port 8000

# テスト実行
python qwen_app_all_in_one.py --test-llamaindex
```

---

## 🔧 オプション: ModelicaでGRAG構築

```python
# your_grag.py
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

# Modelicaファイルを読み込み
from pathlib import Path
docs = []
for mo_file in Path("./Modelica").rglob("*.mo"):
    docs.append(Document(text=mo_file.read_text()))

# インデックス構築
index = VectorStoreIndex.from_documents(docs)

# 質問
query_engine = index.as_query_engine()
response = query_engine.query("Resistorの実装例は？")
print(response.response)
```

実行:
```bash
python your_grag.py
```

---

## 📊 期待される結果

### 初回起動（モデルダウンロード込み）:
- 時間: 10-15分
- ダウンロード: 約13GB

### 2回目以降:
- 起動時間: 30秒
- 応答速度: 3-5秒/100トークン

---

## 🎯 チェックリスト

- [ ] config.json 作成
- [ ] パッケージインストール完了
- [ ] APIサーバー起動成功
- [ ] http://localhost:8000 でWeb UI確認
- [ ] チャットで「こんにちは」と送信して応答確認
- [ ] （オプション）LlamaIndexテスト実行

すべてチェックが付いたら **完了！** 🎉

---

## 🐛 トラブル時の確認

```bash
# ヘルスチェック
curl http://localhost:8000/health

# 期待される応答:
# {"status":"ok","model":"Qwen/Qwen2.5-Coder-7B-Instruct"}
```

---

## 💡 次のステップ

1. ✅ Web UIでいろいろ質問してみる
2. ✅ config.jsonのtemperatureを変えて挙動を確認
3. ✅ 実際のModelicaファイルでGRAG構築
4. ✅ コード生成 → 検証ループの実装

**これで完全なローカルLLMシステムが完成！** 🚀

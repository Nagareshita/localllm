# 💬 チャットモード vs 単純LLMモード

## 設定方法（config.json）

### オプション1: チャットモード（デフォルト、推奨）

```json
{
  "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "use_chat_template": true,
  "system": "You are a helpful coding assistant.",
  ...
}
```

### オプション2: 単純LLMモード（プレーンテキスト）

```json
{
  "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "use_chat_template": false,
  "system": "",
  ...
}
```

---

## 🔍 違いの詳細

### チャットモード（`use_chat_template: true`）

**入力:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"}
  ]
}
```

**内部処理:**
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
```

**特徴:**
- ✅ モデルが学習した形式で推論（最高品質）
- ✅ 会話コンテキストを正しく理解
- ✅ role（system/user/assistant）を明確に区別
- ✅ Instructモデルに最適

**用途:**
- 対話型アプリケーション
- チャットボット
- マルチターン会話
- LlamaIndexとの統合

---

### 単純LLMモード（`use_chat_template: false`）

**入力:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"}
  ]
}
```

**内部処理:**
```
You are a helpful assistant.

Hello
```

**特徴:**
- ✅ プレーンテキスト生成
- ✅ 特殊トークンなし
- ✅ ベースモデルと互換性あり
- ⚠️ 会話構造の理解が弱い

**用途:**
- テキスト補完
- コード補完
- 単発の生成タスク
- ベースモデル使用時

---

## 📊 パフォーマンス比較

| 項目 | チャットモード | 単純LLMモード |
|------|--------------|--------------|
| **応答品質** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **会話理解** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **マルチターン** | ⭐⭐⭐⭐⭐ | ⭐ |
| **単発生成** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **トークン効率** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🎯 使い分けガイド

### チャットモードを使うべき場合

✅ Instructモデルを使用（`-Instruct`が名前に含まれる）
✅ 対話型アプリケーション
✅ LlamaIndexでRAG構築
✅ 複数ターンの会話
✅ システムプロンプトを活用したい

**例:**
- Qwen2.5-Coder-7B-**Instruct** ← これ
- Llama-3-8B-**Instruct**
- Mistral-7B-**Instruct**

### 単純LLMモードを使うべき場合

✅ ベースモデルを使用（Instructではない）
✅ コード補完
✅ テキスト補完
✅ トークン数を最小化したい
✅ カスタムフォーマットを使いたい

**例:**
- Qwen2.5-Coder-7B（Instructなし）
- CodeLlama-7B（Instructなし）
- GPT-2

---

## 💡 実例

### 例1: チャットモード（推奨）

**config.json:**
```json
{
  "use_chat_template": true,
  "system": "You are a Modelica expert."
}
```

**リクエスト:**
```python
messages = [
    {"role": "user", "content": "Resistorの実装例は？"}
]
```

**応答:**
```
Resistorの基本的な実装例を示します：

model Resistor "理想抵抗"
  extends Modelica.Electrical.Analog.Interfaces.OnePort;
  parameter Modelica.SIunits.Resistance R=1 "抵抗値";
equation
  v = R*i;
end Resistor;
```

### 例2: 単純LLMモード

**config.json:**
```json
{
  "use_chat_template": false,
  "system": ""
}
```

**リクエスト:**
```python
messages = [
    {"role": "user", "content": "model Resistor"}
]
```

**応答:**
```
model Resistor "Ideal resistor"
  extends OnePort;
  parameter Real R=1;
equation
  v = R*i;
end Resistor;
```
（systemプロンプトなしなので、日本語指示がない）

---

## 🔧 CLI・APIでの動作

### CLIモード

```bash
# チャットモード
python qwen_app_all_in_one.py --config config.json
> こんにちは
こんにちは！何かお手伝いできることはありますか？

# 単純LLMモード（use_chat_template: false）
python qwen_app_all_in_one.py --config config.json
> こんにちは
（プレーンテキスト生成、会話性が弱い）
```

### APIモード

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

→ `use_chat_template`の設定に従って処理

---

## 📝 設定例集

### 設定1: 対話型チャットボット（推奨）

```json
{
  "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "use_chat_template": true,
  "system": "You are a friendly and helpful assistant.",
  "temperature": 0.7,
  "top_p": 0.9
}
```

### 設定2: Modelica専門家

```json
{
  "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "use_chat_template": true,
  "system": "You are a Modelica modeling expert. Provide accurate code examples.",
  "temperature": 0.3,
  "top_p": 0.9
}
```

### 設定3: コード補完（決定的）

```json
{
  "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "use_chat_template": true,
  "system": "Complete the code without explanation.",
  "temperature": 0.1,
  "top_p": 0.95,
  "do_sample": false
}
```

### 設定4: プレーンテキスト生成

```json
{
  "model_id": "Qwen/Qwen2.5-Coder-7B",
  "use_chat_template": false,
  "system": "",
  "temperature": 0.7,
  "top_p": 0.9
}
```

---

## ⚙️ LlamaIndexでの使い分け

### チャットモード（推奨）

```python
from llama_index.llms.openai_like import OpenAILike

Settings.llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key="dummy",
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    is_chat_model=True,  # ← チャットモード
)
```

### 単純LLMモード

```python
Settings.llm = OpenAILike(
    api_base="http://localhost:8000/v1",
    api_key="dummy",
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    is_chat_model=False,  # ← 補完モード
)
```

**注:** `is_chat_model`はLlamaIndex側の設定で、サーバー側の`use_chat_template`とは独立しています。通常は両方ともtrueに設定します。

---

## 🎯 推奨設定

**99%のケースでチャットモードを推奨します！**

```json
{
  "use_chat_template": true,
  "system": "適切なシステムプロンプト"
}
```

理由:
- ✅ Instructモデルの性能を最大化
- ✅ 会話の文脈を正しく理解
- ✅ LlamaIndexとの相性が良い
- ✅ Web UIで自然な対話

---

## 🔄 切り替え方法

1. **config.jsonを編集**
   ```json
   "use_chat_template": false  // true → false
   ```

2. **APIサーバーを再起動**
   ```bash
   # Ctrl+C で停止
   uvicorn qwen_app_all_in_one:app --host 0.0.0.0 --port 8000
   ```

3. **動作確認**
   - Web UIやCLIでテスト
   - 応答の違いを確認

---

## ✅ まとめ

| 用途 | `use_chat_template` | `system` | 推奨 |
|------|-------------------|----------|------|
| **チャットボット** | `true` | 設定する | ⭐⭐⭐⭐⭐ |
| **LlamaIndex RAG** | `true` | 設定する | ⭐⭐⭐⭐⭐ |
| **対話型アプリ** | `true` | 設定する | ⭐⭐⭐⭐⭐ |
| **コード補完** | `false` | 空 | ⭐⭐⭐ |
| **ベースモデル** | `false` | 空 | ⭐⭐⭐ |

**迷ったらチャットモード（`true`）を選んでください！** 🎉

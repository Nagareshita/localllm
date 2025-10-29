# qwen_app_all_in_one.py
# =====================================================
# 🔥 完全統合版: CLI + API + LlamaIndex対応 + Web UI
# =====================================================
#
# 使い方:
# 1. CLI対話モード:
#    python qwen_app_all_in_one.py --config config.json
#
# 2. 1回だけ生成:
#    python qwen_app_all_in_one.py --config config.json --once "質問"
#
# 3. APIサーバー起動（WSL2推奨）:
#    uvicorn qwen_app_all_in_one:app --host 0.0.0.0 --port 8000
#    → Windowsブラウザから http://localhost:8000 でWeb UIにアクセス
#    → LlamaIndexから http://localhost:8000/v1/chat/completions で利用
#
# 4. LlamaIndexテスト（同じファイル内に例を含む）:
#    python qwen_app_all_in_one.py --test-llamaindex

from __future__ import annotations
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

try:
    from transformers import BitsAndBytesConfig
    HAVE_BNB = True
except Exception:
    HAVE_BNB = False

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    HAVE_FASTAPI = True
except Exception:
    HAVE_FASTAPI = False

# ====== ユーティリティ ======
def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def to_dtype(s: str):
    m = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    return m.get(s, torch.bfloat16)

def repo_to_local_dir(model_id: str) -> Path:
    name = model_id.split("/")[-1]
    return Path.cwd() / "models" / name

def has_safetensors(model_dir: Path) -> bool:
    return any(model_dir.glob("**/*.safetensors"))

# ====== モデル取得 ======
def ensure_model_available(model_id: str, model_dir: Path, offline: bool):
    if has_safetensors(model_dir):
        return
    if offline:
        raise FileNotFoundError(f"[OFFLINE] モデルが見つかりません: {model_dir}")
    
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError("huggingface-hub が必要です。`pip install huggingface_hub`") from e
    
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] ダウンロード開始: {model_id} -> {model_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.py", "tokenizer.*"],
        ignore_patterns=["*.onnx", "*.gguf", "*.md"],
    )

# ====== StoppingCriteria ======
class StopOnSequences(StoppingCriteria):
    def __init__(self, stop_ids: List[List[int]]):
        self.stop_ids = stop_ids
        self.max_len = max((len(s) for s in stop_ids), default=0)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        if not self.stop_ids:
            return False
        seq = input_ids[0].tolist()
        tail = seq[-self.max_len:] if self.max_len else []
        return any(len(tail) >= len(s) and tail[-len(s):] == s for s in self.stop_ids)

# ====== QwenRunner ======
class QwenRunner:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.model_id = cfg.get("model_id", "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.model_dir = repo_to_local_dir(self.model_id)
        self.offline = os.environ.get("TRANSFORMERS_OFFLINE", "") == "1"
        
        ensure_model_available(self.model_id, self.model_dir, self.offline)
        
        dtype = to_dtype(cfg.get("torch_dtype", "bfloat16"))
        device_map = cfg.get("device_map", "auto")
        
        quant = str(cfg.get("quantization", "")).lower().strip()
        bnb_config = None
        if quant in {"bnb-4bit", "bnb_4bit", "4bit"}:
            if not HAVE_BNB:
                raise RuntimeError("bitsandbytes が必要です")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=to_dtype(cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
                bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
                bnb_4bit_quant_type=str(cfg.get("bnb_4bit_quant_type", "nf4")),
            )
        elif quant in {"bnb-8bit", "bnb_8bit", "8bit"}:
            if not HAVE_BNB:
                raise RuntimeError("bitsandbytes が必要です")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            torch_dtype=(None if bnb_config else dtype),
            device_map=device_map,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        self.model.eval()
        
        self.eos_id = cfg.get("eos_token_id", getattr(self.tokenizer, "eos_token_id", None))
        self.pad_id = cfg.get("pad_token_id", getattr(self.tokenizer, "pad_token_id", None) or self.eos_id)
    
    def _encode_messages(self, messages: List[Dict[str, str]]):
        """OpenAI互換のmessages形式からエンコード"""
        if self.cfg.get("use_chat_template", True):
            # チャットテンプレート使用
            chat_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            enc = self.tokenizer(chat_text, return_tensors="pt")
        else:
            # 単純なLLMモード（チャットテンプレートなし）
            # systemメッセージとuserメッセージを連結
            text_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system" and content:
                    text_parts.append(content)
                elif role == "user":
                    text_parts.append(content)
            
            full_text = "\n\n".join(text_parts)
            enc = self.tokenizer(full_text, return_tensors="pt", truncation=True)
        
        return {k: v.to(self.model.device) for k, v in enc.items()}
    
    def generate_from_messages(self, messages: List[Dict[str, str]], **override_params) -> Dict[str, Any]:
        inputs = self._encode_messages(messages)
        input_len = inputs["input_ids"].shape[-1]
        
        gen_kwargs = {
            "max_new_tokens": override_params.get("max_tokens", self.cfg.get("max_new_tokens", 512)),
            "temperature": override_params.get("temperature", self.cfg.get("temperature", 0.7)),
            "top_p": override_params.get("top_p", self.cfg.get("top_p", 0.9)),
            "repetition_penalty": self.cfg.get("repetition_penalty", 1.05),
            "do_sample": True if override_params.get("temperature", 0.7) > 0 else False,
            "use_cache": True,
            "eos_token_id": self.eos_id,
            "pad_token_id": self.pad_id,
        }
        
        if "top_k" in override_params or self.cfg.get("top_k", 0):
            gen_kwargs["top_k"] = int(override_params.get("top_k", self.cfg.get("top_k", 0)))
        
        stop = override_params.get("stop", self.cfg.get("stop_words", []))
        if stop:
            if isinstance(stop, str):
                stop = [stop]
            stop_ids = [self.tokenizer.encode(s, add_special_tokens=False) for s in stop]
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([StopOnSequences(stop_ids)])
        
        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        
        new_tokens = out[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return {
            "text": text,
            "prompt_tokens": int(input_len),
            "completion_tokens": int(new_tokens.shape[-1]),
            "total_tokens": int(out.shape[-1]),
        }

# ====== CLI ======
def run_cli(cfg_path: Path, once: str | None):
    cfg = read_json(cfg_path)
    runner = QwenRunner(cfg)
    system = cfg.get("system", "You are a helpful assistant.")
    
    if once is not None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": once})
        print(runner.generate_from_messages(messages)["text"])
        return
    
    print("Qwen CLI Chat (Ctrl+C で終了)")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    
    while True:
        try:
            q = input("> ")
            messages.append({"role": "user", "content": q})
            result = runner.generate_from_messages(messages)
            answer = result["text"]
            print(answer)
            messages.append({"role": "assistant", "content": answer})
        except KeyboardInterrupt:
            print("\nbye.")
            break

# ====== FastAPI ======
if HAVE_FASTAPI:
    class ChatMessage(BaseModel):
        role: str
        content: str
    
    class ChatCompletionRequest(BaseModel):
        model: str
        messages: List[ChatMessage]
        temperature: Optional[float] = 0.7
        top_p: Optional[float] = 0.9
        top_k: Optional[int] = None
        max_tokens: Optional[int] = 512
        stop: Optional[List[str] | str] = None
        stream: Optional[bool] = False
    
    class ChatCompletionChoice(BaseModel):
        index: int
        message: ChatMessage
        finish_reason: str
    
    class ChatCompletionUsage(BaseModel):
        prompt_tokens: int
        completion_tokens: int
        total_tokens: int
    
    class ChatCompletionResponse(BaseModel):
        id: str
        object: str = "chat.completion"
        created: int
        model: str
        choices: List[ChatCompletionChoice]
        usage: ChatCompletionUsage
    
    _CFG_PATH_ENV = os.environ.get("QWEN_CFG_PATH", "")
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg_path = Path(_CFG_PATH_ENV) if _CFG_PATH_ENV else Path.cwd() / "config.json"
        if not cfg_path.exists():
            raise RuntimeError(f"設定ファイルが見つかりません: {cfg_path}")
        app.state.cfg = read_json(cfg_path)
        app.state.runner = QwenRunner(app.state.cfg)
        print(f"✅ モデル: {app.state.runner.model_id}")
        print(f"✅ デバイス: {app.state.runner.model.device}")
        try:
            yield
        finally:
            if hasattr(app.state, "runner"):
                del app.state.runner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    app = FastAPI(title="Qwen All-in-One API", lifespan=lifespan)
    
    try:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    except Exception:
        pass
    
    # OpenAI互換エンドポイント
    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(req: ChatCompletionRequest):
        if req.stream:
            return {"error": "Streaming not supported"}, 400
        
        runner: QwenRunner = app.state.runner
        messages = [{"role": msg.role, "content": msg.content} for msg in req.messages]
        
        override_params = {}
        if req.temperature is not None: override_params["temperature"] = req.temperature
        if req.top_p is not None: override_params["top_p"] = req.top_p
        if req.top_k is not None: override_params["top_k"] = req.top_k
        if req.max_tokens is not None: override_params["max_tokens"] = req.max_tokens
        if req.stop is not None: override_params["stop"] = req.stop
        
        result = runner.generate_from_messages(messages, **override_params)
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=result["text"]),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                total_tokens=result["total_tokens"]
            )
        )
    
    @app.get("/health")
    def health():
        return {"status": "ok", "model": app.state.runner.model_id}
    
    # Web UI用シンプルチャットエンドポイント
    @app.post("/chat")
    def simple_chat(req: dict):
        runner: QwenRunner = app.state.runner
        user_message = req.get("message", "")
        history = req.get("history", [])
        
        messages = []
        system = app.state.cfg.get("system", "You are a helpful assistant.")
        if system:
            messages.append({"role": "system", "content": system})
        
        for h in history:
            messages.append({"role": "user", "content": h["user"]})
            messages.append({"role": "assistant", "content": h["assistant"]})
        
        messages.append({"role": "user", "content": user_message})
        
        result = runner.generate_from_messages(messages)
        return {"response": result["text"], "usage": {
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"]
        }}
    
    # Web UI HTMLを返す
    @app.get("/", response_class=HTMLResponse)
    def web_ui():
        return """<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Qwen Local Chat</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .message {
            margin-bottom: 15px;
            display: flex;
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .message.user { justify-content: flex-end; }
        .message.assistant { justify-content: flex-start; }
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .message.assistant .message-content {
            background: white;
            color: #333;
            border: 1px solid #ddd;
        }
        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #ddd;
            display: flex;
            gap: 10px;
        }
        #userInput {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #667eea;
            border-radius: 25px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
        }
        #userInput:focus { border-color: #764ba2; }
        #sendBtn {
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        #sendBtn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4); }
        #sendBtn:active { transform: translateY(0); }
        #sendBtn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .loading {
            display: none;
            text-align: center;
            color: #667eea;
            padding: 10px;
            font-style: italic;
        }
        .stats {
            font-size: 11px;
            color: #999;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">🤖 Qwen Local Chat</div>
        <div class="chat-container" id="chatContainer">
            <div class="message assistant">
                <div class="message-content">こんにちは！何でも聞いてください。</div>
            </div>
        </div>
        <div class="loading" id="loading">生成中...</div>
        <div class="input-container">
            <input type="text" id="userInput" placeholder="メッセージを入力..." />
            <button id="sendBtn">送信</button>
        </div>
    </div>
    <script>
        const chatContainer = document.getElementById('chatContainer');
        const userInput = document.getElementById('userInput');
        const sendBtn = document.getElementById('sendBtn');
        const loading = document.getElementById('loading');
        let history = [];

        function addMessage(role, content, stats = null) {
            const msg = document.createElement('div');
            msg.className = `message ${role}`;
            
            const msgContent = document.createElement('div');
            msgContent.className = 'message-content';
            msgContent.textContent = content;
            
            if (stats) {
                const statsDiv = document.createElement('div');
                statsDiv.className = 'stats';
                statsDiv.textContent = `トークン: ${stats.prompt_tokens} + ${stats.completion_tokens} = ${stats.prompt_tokens + stats.completion_tokens}`;
                msgContent.appendChild(statsDiv);
            }
            
            msg.appendChild(msgContent);
            chatContainer.appendChild(msg);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message) return;
            
            addMessage('user', message);
            userInput.value = '';
            sendBtn.disabled = true;
            loading.style.display = 'block';
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message, history })
                });
                
                const data = await response.json();
                addMessage('assistant', data.response, data.usage);
                
                history.push({ user: message, assistant: data.response });
                if (history.length > 10) history.shift();  // 履歴は最大10ターン
            } catch (error) {
                addMessage('assistant', 'エラーが発生しました: ' + error.message);
            } finally {
                loading.style.display = 'none';
                sendBtn.disabled = false;
                userInput.focus();
            }
        }

        sendBtn.addEventListener('click', sendMessage);
        userInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        userInput.focus();
    </script>
</body>
</html>"""

# ====== LlamaIndexテスト ======
def test_llamaindex():
    """LlamaIndexとの統合テスト（同じファイル内で完結）"""
    print("🧪 LlamaIndexテストを開始...")
    
    try:
        from llama_index.core import VectorStoreIndex, Document, Settings
        from llama_index.llms.openai_like import OpenAILike
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as e:
        print(f"❌ LlamaIndexが未インストールです: {e}")
        print("   pip install llama-index llama-index-llms-openai-like llama-index-embeddings-huggingface")
        return
    
    # ローカルLLM設定
    print("⚙️  LlamaIndex設定中...")
    Settings.llm = OpenAILike(
        api_base="http://localhost:8000/v1",
        api_key="dummy",
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        temperature=0.7,
        max_tokens=200,
        timeout=60,
        is_chat_model=True,
    )
    
    # 軽量埋め込みモデル
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./models/embeddings"
    )
    
    # サンプルドキュメント
    documents = [
        Document(
            text="model Resistor is a basic electrical component with resistance R.",
            metadata={"type": "component"}
        ),
        Document(
            text="model Capacitor stores electrical energy with capacitance C.",
            metadata={"type": "component"}
        ),
    ]
    
    print("🔨 インデックス構築中...")
    index = VectorStoreIndex.from_documents(documents)
    
    print("🔍 クエリ実行中...")
    query_engine = index.as_query_engine()
    response = query_engine.query("What is a resistor?")
    
    print("\n✅ LlamaIndexテスト成功!")
    print(f"質問: What is a resistor?")
    print(f"回答: {response.response}")
    print("\n💡 この方法でModelicaのGRAGを構築できます！")

# ====== エントリーポイント ======
def main():
    ap = argparse.ArgumentParser(description="Qwen All-in-One: CLI + API + LlamaIndex")
    ap.add_argument("--config", default="config.json", help="設定JSON")
    ap.add_argument("--once", default=None, help="1回だけ生成")
    ap.add_argument("--test-llamaindex", action="store_true", help="LlamaIndex統合テスト")
    args = ap.parse_args()
    
    if args.test_llamaindex:
        test_llamaindex()
        return
    
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"❌ 設定ファイルが見つかりません: {cfg_path}")
        print("   config.json を作成してください")
        sys.exit(1)
    
    # 環境変数に設定ファイルパスを保存（uvicorn起動時用）
    os.environ["QWEN_CFG_PATH"] = str(cfg_path)
    
    # CLIモード
    run_cli(cfg_path, args.once)

if __name__ == "__main__":
    main()

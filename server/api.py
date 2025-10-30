from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import os
import time
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from .runner import QwenRunner, read_json

try:
    from .li_memory import LIMemoryManager, LIMemoryConfig
    HAVE_LI_MEMORY = True
except Exception:
    HAVE_LI_MEMORY = False


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

# ====== セッション永続化 ======
from threading import Lock
from datetime import datetime
import uuid

_SESS_DIR = Path.cwd() / "sessions"
_SESS_DIR.mkdir(parents=True, exist_ok=True)
_SESS_LOCK = Lock()


def _sess_path(sid: str) -> Path:
    return _SESS_DIR / f"{sid}.json"


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _list_sessions_meta() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for p in sorted(_SESS_DIR.glob("*.json")):
        try:
            data = read_json(p)
            items.append({
                "id": data.get("id", p.stem),
                "title": data.get("title", p.stem),
                "created": data.get("created", ""),
                "updated": data.get("updated", ""),
                "length": len(data.get("history", [])),
            })
        except Exception:
            continue
    items.sort(key=lambda x: x.get("updated", ""), reverse=True)
    return items


def _load_session(sid: str) -> Dict[str, Any]:
    p = _sess_path(sid)
    if not p.exists():
        raise FileNotFoundError("session not found")
    return read_json(p)


def _save_session(doc: Dict[str, Any]) -> None:
    p = _sess_path(doc["id"])
    with _SESS_LOCK:
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 既定は server/config.json を優先、環境変数があればそちらを使用
    if _CFG_PATH_ENV:
        cfg_path = Path(_CFG_PATH_ENV)
    else:
        default_server_cfg = Path.cwd() / "server" / "config.json"
        default_root_cfg = Path.cwd() / "config.json"
        cfg_path = default_server_cfg if default_server_cfg.exists() else default_root_cfg
    if not cfg_path.exists():
        raise RuntimeError(f"設定ファイルが見つかりません: {cfg_path}")
    app.state.cfg = read_json(cfg_path)
    app.state.runner = QwenRunner(app.state.cfg)
    print(f"🧠 モデル: {app.state.runner.model_id}")
    print(f"💻 デバイス: {app.state.runner.model.device}")
    if HAVE_LI_MEMORY:
        try:
            app.state.memory = LIMemoryManager(
                LIMemoryConfig(
                    api_base="http://localhost:8003/v1-nomem",
                    model=app.state.runner.model_id,
                    short_turn_threshold=10,
                    vector_every_n_turns=3,
                )
            )
            print("🗂️ メモリ機能: 有効")
        except Exception as e:
            print(f"⚠️ メモリ初期化失敗: {e}")
            app.state.memory = None
    else:
        app.state.memory = None
    try:
        yield
    finally:
        if hasattr(app.state, "runner"):
            del app.state.runner
        if hasattr(app.state, "memory"):
            del app.state.memory


app = FastAPI(title="Qwen Local API", lifespan=lifespan)

# CORS: Windows側のfile://やhttp://localhostから許可
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


@app.get("/health")
def health():
    return {"status": "ok", "model": app.state.runner.model_id}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        return {"error": "Streaming not supported"}, 400

    runner: QwenRunner = app.state.runner
    memory = getattr(app.state, "memory", None)

    base_messages = [{"role": m.role, "content": m.content} for m in req.messages]
    last_user = next((m for m in reversed(base_messages) if m.get("role") == "user"), None)
    user_query = last_user.get("content") if last_user else ""
    messages = list(base_messages)

    if memory and user_query:
        try:
            mem_ctx = memory.build_context(user_query)
        except Exception:
            mem_ctx = ""
        if mem_ctx:
            try:
                last_user_idx = max(i for i, m in enumerate(messages) if m.get("role") == "user")
            except ValueError:
                last_user_idx = len(messages)
            messages.insert(last_user_idx, {"role": "system", "content": f"Relevant memory:\n{mem_ctx}"})

    override_params: Dict[str, Any] = {}
    if req.temperature is not None: override_params["temperature"] = req.temperature
    if req.top_p is not None: override_params["top_p"] = req.top_p
    if req.top_k is not None: override_params["top_k"] = req.top_k
    if req.max_tokens is not None: override_params["max_tokens"] = req.max_tokens
    if req.stop is not None: override_params["stop"] = req.stop

    result = runner.generate_from_messages(messages, **override_params)
    answer = result["text"]

    if memory and user_query:
        try:
            memory.record_turn(user_query, answer)
        except Exception:
            pass

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=req.model,
        choices=[ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=answer), finish_reason="stop")],
        usage=ChatCompletionUsage(
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("completion_tokens", 0),
            total_tokens=result.get("total_tokens", 0),
        ),
    )


@app.post("/v1-nomem/chat/completions", response_model=ChatCompletionResponse)
def chat_completions_nomem(req: ChatCompletionRequest):
    if req.stream:
        return {"error": "Streaming not supported"}, 400

    runner: QwenRunner = app.state.runner
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    override_params: Dict[str, Any] = {}
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
        choices=[ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=result["text"]), finish_reason="stop")],
        usage=ChatCompletionUsage(
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("completion_tokens", 0),
            total_tokens=result.get("total_tokens", 0),
        ),
    )


# ====== セッションAPI ======
@app.get("/api/sessions")
def api_sessions_list():
    return {"sessions": _list_sessions_meta()}


@app.post("/api/sessions")
def api_sessions_create(req: Dict[str, Any]):
    sid = uuid.uuid4().hex[:8]
    title = req.get("title") if isinstance(req, dict) else None
    if not title:
        title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    doc = {
        "id": sid,
        "title": title,
        "created": _now_iso(),
        "updated": _now_iso(),
        "history": [],
    }
    _save_session(doc)
    return doc


@app.get("/api/sessions/{sid}")
def api_sessions_get(sid: str):
    try:
        doc = _load_session(sid)
        return doc
    except Exception:
        return {"error": "not found"}, 404


@app.post("/api/sessions/{sid}/chat")
def api_sessions_chat(sid: str, req: Dict[str, Any]):
    try:
        doc = _load_session(sid)
    except Exception:
        return {"error": "not found"}, 404

    runner: QwenRunner = app.state.runner
    memory = getattr(app.state, "memory", None)

    user_message = str(req.get("message", ""))
    system = app.state.cfg.get("system", "You are a helpful assistant.")

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    for h in doc.get("history", []):
        messages.append({"role": "user", "content": h.get("user", "")})
        messages.append({"role": "assistant", "content": h.get("assistant", "")})
    messages.append({"role": "user", "content": user_message})

    if memory and user_message:
        try:
            mem_ctx = memory.build_context(user_message)
        except Exception:
            mem_ctx = ""
        if mem_ctx:
            try:
                last_user_idx = max(i for i, m in enumerate(messages) if m.get("role") == "user")
            except ValueError:
                last_user_idx = len(messages)
            messages.insert(last_user_idx, {"role": "system", "content": f"Relevant memory:\n{mem_ctx}"})

    result = runner.generate_from_messages(messages)
    answer = result["text"]

    doc["history"].append({"user": user_message, "assistant": answer})
    doc["updated"] = _now_iso()
    _save_session(doc)

    if memory and user_message:
        try:
            memory.record_turn(user_message, answer)
        except Exception:
            pass

    return {"response": answer, "usage": {
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0)
    }}

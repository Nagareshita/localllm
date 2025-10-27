# qwen_app.py
# - API:   uvicorn qwen_app:app --host 127.0.0.1 --port 8000
# - CLI:   python qwen_app.py --config config.json           （対話）
#         python qwen_app.py --config config.json --once "プロンプト"（1回だけ）
#
# 仕様:
# - ./models/<repo_name>/ にモデルを保存（なければダウンロード、あればローカル使用）
# - TRANSFORMERS_OFFLINE=1 の時はダウンロードしない（ローカルのみ使用）
# - JSON設定で各種パラメータを制御
# - Windows対応（Pathlib使用）

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List
from contextlib import asynccontextmanager

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
try:
    from transformers import BitsAndBytesConfig  # optional (量子化)
    HAVE_BNB = True
except Exception:
    HAVE_BNB = False

# --- FastAPI（APIモードでのみ必要） ---
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
    HAVE_FASTAPI = True
except Exception:
    HAVE_FASTAPI = False

# ====== ユーティリティ ======
def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def to_dtype(s: str):
    m = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    return m.get(s, torch.bfloat16)

def repo_to_local_dir(model_id: str) -> Path:
    # 例: "Qwen/Qwen2.5-Coder-7B-Instruct" -> ./models/Qwen2.5-Coder-7B-Instruct
    name = model_id.split("/")[-1]
    return Path.cwd() / "models" / name

def has_safetensors(model_dir: Path) -> bool:
    return any(model_dir.glob("**/*.safetensors"))

# ====== モデル取得（必要ならDL） ======
def ensure_model_available(model_id: str, model_dir: Path, offline: bool):
    """
    - offline=True: ローカルのみ。存在しなければ明示エラー。
    - offline=False: なければHugging Faceからダウンロード。
    """
    if has_safetensors(model_dir):
        return  # 既にOK

    if offline:
        raise FileNotFoundError(
            f"[OFFLINE] モデルが見つかりません: {model_dir}\n"
            "事前にオンラインで一度ダウンロードするか、モデルフォルダに手動配置してください。"
        )

    # オンラインで取得
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "huggingface-hub が必要です。`pip install huggingface_hub` を実行してください。"
        ) from e

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] ダウンロード開始: {model_id} -> {model_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,  # Windows配慮
        allow_patterns=[
            "*.safetensors", "*.json", "*.model", "*.txt", "*.py",
            "tokenizer.*", "tokenizer_*", "special_tokens_map.json",
            "generation_config.json", "config.json"
        ],
        ignore_patterns=["*.onnx", "*.gguf", "*.md"],
        revision=None
    )
    if not has_safetensors(model_dir):
        raise RuntimeError("ダウンロード後も .safetensors が見つかりません。取得失敗の可能性があります。")

# ====== StoppingCriteria（任意の停止語） ======
class StopOnSequences(StoppingCriteria):
    def __init__(self, stop_ids: List[List[int]]):
        self.stop_ids = stop_ids
        self.max_len = max((len(s) for s in stop_ids), default=0)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        if not self.stop_ids:
            return False
        seq = input_ids[0].tolist()
        tail = seq[-self.max_len:] if self.max_len else []
        for s in self.stop_ids:
            if len(tail) >= len(s) and tail[-len(s):] == s:
                return True
        return False

# ====== ロードと推論 ======
class QwenRunner:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.model_id = cfg.get("model_id", "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.model_dir = repo_to_local_dir(self.model_id)
        self.offline = os.environ.get("TRANSFORMERS_OFFLINE", "") == "1"

        # モデルを確保（ローカル or DL）
        ensure_model_available(self.model_id, self.model_dir, self.offline)

        # ローカルから読み込む
        dtype = to_dtype(cfg.get("torch_dtype", "bfloat16"))
        device_map = cfg.get("device_map", "auto")

        # 量子化オプション（bitsandbytes）
        quant = str(cfg.get("quantization", "")).lower().strip()
        bnb_config = None
        if quant in {"bnb-4bit", "bnb_4bit", "4bit"}:
            if not HAVE_BNB:
                raise RuntimeError(
                    "bitsandbytes が必要です。WSL/Linuxでは `pip install bitsandbytes`、"
                    "Windowsネイティブでは `pip install bitsandbytes-windows==0.43.1` を検討してください。"
                )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=to_dtype(cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
                bnb_4bit_use_double_quant=bool(cfg.get("bnb_4bit_use_double_quant", True)),
                bnb_4bit_quant_type=str(cfg.get("bnb_4bit_quant_type", "nf4")),
            )
        elif quant in {"bnb-8bit", "bnb_8bit", "8bit"}:
            if not HAVE_BNB:
                raise RuntimeError(
                    "bitsandbytes が必要です。WSL/Linuxでは `pip install bitsandbytes`、"
                    "Windowsネイティブでは `pip install bitsandbytes-windows==0.43.1` を検討してください。"
                )
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_dir),
            torch_dtype=(None if bnb_config else dtype),
            device_map=device_map,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        self.model.eval()

        # eos/pad
        self.eos_id = cfg.get("eos_token_id", getattr(self.tokenizer, "eos_token_id", None))
        self.pad_id = cfg.get("pad_token_id", getattr(self.tokenizer, "pad_token_id", None) or self.eos_id)

    def _encode(self, prompt: str):
        if self.cfg.get("use_chat_template", True):
            system = self.cfg.get("system", "You are a helpful coding assistant.")
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt})
            # 文字列でテンプレートを生成し、その後に通常のトークナイズを行う
            chat_text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            enc = self.tokenizer(chat_text, return_tensors="pt")
        else:
            enc = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        return {k: v.to(self.model.device) for k, v in enc.items()}

    def generate(self, prompt: str) -> Dict[str, Any]:
        inputs = self._encode(prompt)
        input_len = inputs["input_ids"].shape[-1]

        # 停止語（任意）
        stop_words = self.cfg.get("stop_words", [])
        stop_ids = [self.tokenizer.encode(s, add_special_tokens=False) for s in stop_words] if stop_words else []
        stopping = StoppingCriteriaList([StopOnSequences(stop_ids)]) if stop_ids else None

        gen_kwargs = {
            "max_new_tokens": self.cfg.get("max_new_tokens", 512),
            "min_new_tokens": self.cfg.get("min_new_tokens", 0),
            "temperature": self.cfg.get("temperature", 0.7),
            "top_p": self.cfg.get("top_p", 0.9),
            "repetition_penalty": self.cfg.get("repetition_penalty", 1.05),
            "no_repeat_ngram_size": self.cfg.get("no_repeat_ngram_size", 0),
            "length_penalty": self.cfg.get("length_penalty", 1.0),
            "num_beams": self.cfg.get("num_beams", 1),
            "early_stopping": self.cfg.get("early_stopping", False),
            "do_sample": self.cfg.get("do_sample", True),
            "use_cache": True,
            "eos_token_id": self.eos_id,
            "pad_token_id": self.pad_id,
        }
        if self.cfg.get("top_k", 0):
            gen_kwargs["top_k"] = int(self.cfg["top_k"])
        if stopping:
            gen_kwargs["stopping_criteria"] = stopping

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        # 生成部分のみ
        new_tokens = out[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "model": self.model_id,
            "result": text,
            "usage": {
                "prompt_tokens": int(input_len),
                "completion_tokens": int(new_tokens.shape[-1]),
                "total_tokens": int(out.shape[-1]),
            }
        }

# ====== CLI ======
def run_cli(cfg_path: Path, once: str | None):
    cfg = read_json(cfg_path)
    runner = QwenRunner(cfg)

    if once is not None:
        print(runner.generate(once)["result"])
        return

    print("Qwen Local Chat (Ctrl+C で終了)  / オフライン:", "ON" if runner.offline else "OFF")
    while True:
        try:
            q = input("> ")
            res = runner.generate(q)
            print(res["result"])
        except KeyboardInterrupt:
            print("\nbye.")
            break

# ====== API ======
# FastAPI の依存が無い環境でも CLI は動かしたいので、条件付き定義
if HAVE_FASTAPI:
    class GenReq(BaseModel):
        prompt: str
        temperature: float | None = None
        top_p: float | None = None
        top_k: int | None = None
        max_new_tokens: int | None = None
        do_sample: bool | None = None

    # 起動時に --config を見つけるため、環境変数で受ける
    _CFG_PATH_ENV = os.environ.get("QWEN_CFG_PATH", "")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # startup
        cfg_path = Path(_CFG_PATH_ENV) if _CFG_PATH_ENV else Path.cwd() / "config.json"
        if not cfg_path.exists():
            raise RuntimeError(f"設定ファイルが見つかりません: {cfg_path}")
        app.state.cfg = read_json(cfg_path)
        app.state.runner = QwenRunner(app.state.cfg)
        try:
            yield
        finally:
            # shutdown（必要に応じてリソース解放）
            try:
                if hasattr(app.state, "runner"):
                    del app.state.runner
            except Exception:
                pass
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    app = FastAPI(title="Qwen Local API", lifespan=lifespan)

    # 開発用途の簡易CORS（ブラウザからのアクセス用）
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

    @app.post("/generate")
    def generate(req: GenReq):
        # 共有ランナーを使い回し、生成時だけ一時的にパラメータを上書き
        runner: QwenRunner = app.state.runner
        orig_cfg = runner.cfg
        try:
            cfg = dict(app.state.cfg)
            if req.temperature is not None:   cfg["temperature"] = req.temperature
            if req.top_p is not None:         cfg["top_p"] = req.top_p
            if req.top_k is not None:         cfg["top_k"] = req.top_k
            if req.max_new_tokens is not None: cfg["max_new_tokens"] = req.max_new_tokens
            if req.do_sample is not None:     cfg["do_sample"] = req.do_sample
            runner.cfg = cfg
            return runner.generate(req.prompt)
        finally:
            runner.cfg = orig_cfg

# ====== エントリーポイント ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=False, default="config.json", help="設定JSONのパス")
    ap.add_argument("--once", default=None, help="1回だけ生成して終了（プロンプト文字列）")
    ap.add_argument("--api", action="store_true", help="APIサーバーモードで起動（uvicornが別途必要）")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if args.api:
        if not HAVE_FASTAPI:
            print("FastAPI が未インストールです。`pip install fastapi uvicorn` を実行してください。", file=sys.stderr)
            sys.exit(1)
        # uvicorn から起動される前提。環境変数で設定ファイルパスを渡す
        os.environ["QWEN_CFG_PATH"] = str(cfg_path)
        # API起動は uvicorn から行ってください（コメント参照）
        print("APIモードです。以下で起動してください:")
        print("  uvicorn qwen_app:app --host 127.0.0.1 --port 8000")
        return

    # CLI
    run_cli(cfg_path, args.once)

if __name__ == "__main__":
    main()

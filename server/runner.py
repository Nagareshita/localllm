from __future__ import annotations
from typing import Any, Dict, List, Optional
from pathlib import Path
import os
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

try:
    from transformers import BitsAndBytesConfig
    HAVE_BNB = True
except Exception:
    HAVE_BNB = False


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


def ensure_model_available(model_id: str, model_dir: Path, offline: bool):
    if has_safetensors(model_dir):
        return
    if offline:
        raise FileNotFoundError(f"[OFFLINE] モデルが見つかりません: {model_dir}")
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError("huggingface_hub が必要です。`pip install huggingface_hub`") from e

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] ダウンロード開始: {model_id} -> {model_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.py", "tokenizer.*"],
        ignore_patterns=["*.onnx", "*.gguf", "*.md"],
    )


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
            trust_remote_code=True,
        )
        self.model.eval()

        self.eos_id = self.cfg.get("eos_token_id", getattr(self.tokenizer, "eos_token_id", None))
        self.pad_id = self.cfg.get("pad_token_id", getattr(self.tokenizer, "pad_token_id", None) or self.eos_id)

    def _encode_messages(self, messages: List[Dict[str, str]]):
        if self.cfg.get("use_chat_template", True):
            chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = self.tokenizer(chat_text, return_tensors="pt")
        else:
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


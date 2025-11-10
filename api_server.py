#!/usr/bin/env python3
"""OpenAI互換 API サーバー (Transformers + bitsandbytes)."""

import argparse
import json
import os
import time
import uuid
from pathlib import Path
from threading import Thread
from typing import Iterable, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TextIteratorStreamer)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = None
    max_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    stream: bool = False
    stop: Optional[List[str]] = None


class CompletionRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    stream: bool = False
    stop: Optional[List[str]] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI互換APIサーバー")
    parser.add_argument("--quantization",
                        choices=["8bit", "4bit", "none"],
                        default="8bit",
                        help="量子化方式")
    parser.add_argument("--host", default="0.0.0.0", help="ホスト")
    parser.add_argument("--port", type=int, default=8003, help="ポート")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                        help="Hugging Face モデルID")
    parser.add_argument("--model-dir", default="./models",
                        help="モデルキャッシュディレクトリ")
    # 12GB級GPU向けのCPUオフロード設定（簡単に変更できるようCLI化）
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="8bit時に一部レイヤをCPUへオフロード（VRAMが不足する環境向け）")
    parser.add_argument(
        "--gpu-mem-limit",
        type=str,
        default=None,
        help="GPUごとの上限メモリ（例: 10GiB）。未指定時は12GB級GPUで自動10GiB。")
    parser.add_argument(
        "--cpu-mem-limit",
        type=str,
        default="48GiB",
        help="CPU側の上限メモリ（例: 48GiB）")
    parser.add_argument(
        "--offload-dir",
        type=str,
        default=None,
        help="CPU/ディスクオフロード用の作業ディレクトリ。未指定時は <model-dir>/offload")
    return parser.parse_args()


class LLMEngine:
    """テキスト生成エンジン"""

    def __init__(self,
                 model_id: str,
                 quantization: str,
                 cache_dir: Path,
                 enable_cpu_offload: bool = False,
                 gpu_mem_limit: Optional[str] = None,
                 cpu_mem_limit: Optional[str] = None,
                 offload_dir: Optional[Path] = None):
        self.model_id = model_id
        self.quantization = quantization
        self.cache_dir = cache_dir
        self.enable_cpu_offload = enable_cpu_offload
        self.gpu_mem_limit = gpu_mem_limit
        self.cpu_mem_limit = cpu_mem_limit
        self.offload_dir = offload_dir
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _load_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id,
                                             cache_dir=str(self.cache_dir),
                                             trust_remote_code=True)

    def _load_model(self):
        quant_config = None
        if self.quantization == "8bit":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                # 12GB級GPUではCPUオフロードを有効にすることで起動を安定化
                llm_int8_enable_fp32_cpu_offload=self.enable_cpu_offload,
            )
        elif self.quantization == "4bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=False,
                                              load_in_4bit=True,
                                              bnb_4bit_compute_dtype=torch.bfloat16,
                                              bnb_4bit_quant_type="nf4",
                                              bnb_4bit_use_double_quant=True)

        kwargs = dict(cache_dir=str(self.cache_dir),
                      trust_remote_code=True,
                      device_map="auto")
        if quant_config:
            kwargs["quantization_config"] = quant_config
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        # CPUオフロード設定（8bit時）
        if self.quantization == "8bit" and self.enable_cpu_offload:
            # デフォルトのオフロードフォルダ
            offload_dir = self.offload_dir or (self.cache_dir / "offload")
            offload_dir.mkdir(parents=True, exist_ok=True)

            # GPU上限の推奨既定（12GB級=約10GiBを割当）
            gpu_limit = self.gpu_mem_limit
            if gpu_limit is None:
                try:
                    if torch.cuda.is_available():
                        total = torch.cuda.get_device_properties(0).total_memory
                        # 12GB±の範囲なら10GiB、それ以外は総量の ~80% を上限に設定
                        GiB = 1024 ** 3
                        if 11 * GiB <= total <= 13 * GiB:
                            gpu_limit = "10GiB"
                        else:
                            gpu_limit = f"{int(total * 0.8 / GiB)}GiB"
                except Exception:
                    gpu_limit = "10GiB"

            kwargs["max_memory"] = {0: gpu_limit, "cpu": self.cpu_mem_limit or "48GiB"}
            kwargs["offload_folder"] = str(offload_dir)

        return AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

    def _apply_chat_template(self, messages: List[ChatMessage]) -> str:
        records = [{"role": m.role, "content": m.content} for m in messages]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(records,
                                                      tokenize=False,
                                                      add_generation_prompt=True)
        joined = "".join(f"{m.role}: {m.content}\n" for m in messages)
        return joined + "assistant:"

    def _token_count(self, text: str) -> int:
        encoded = self.tokenizer(text, return_tensors="pt")
        return int(encoded.input_ids.shape[-1])

    def _generation_kwargs(self, max_tokens: int, temperature: float,
                           top_p: float) -> dict:
        do_sample = temperature > 0
        kwargs = dict(max_new_tokens=max_tokens,
                      eos_token_id=self.tokenizer.eos_token_id,
                      pad_token_id=self.tokenizer.pad_token_id,
                      do_sample=do_sample)
        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        return kwargs

    def _trim_stop(self, text: str, stop: Optional[List[str]]):
        if not stop:
            return text, False
        end = None
        for token in stop:
            idx = text.find(token)
            if idx != -1 and (end is None or idx < end):
                end = idx
        if end is None:
            return text, False
        return text[:end], True

    def _prepare_inputs(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_length = inputs.input_ids.shape[-1]
        tensors = {k: v.to(self.model.device) for k, v in inputs.items()}
        return tensors, prompt_length

    def generate(self,
                 prompt: str,
                 max_tokens: int,
                 temperature: float,
                 top_p: float,
                 stop: Optional[List[str]] = None) -> dict:
        inputs, prompt_length = self._prepare_inputs(prompt)
        kwargs = self._generation_kwargs(max_tokens, temperature, top_p)
        output = self.model.generate(**inputs, **kwargs)
        generated = output[0][prompt_length:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        completion, _ = self._trim_stop(text, stop)
        prompt_tokens = self._token_count(prompt)
        completion_tokens = self._token_count(completion) if completion else 0
        return dict(text=completion,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens)

    def stream(self,
               prompt: str,
               max_tokens: int,
               temperature: float,
               top_p: float,
               stop: Optional[List[str]] = None) -> Iterable[str]:
        inputs, prompt_length = self._prepare_inputs(prompt)
        kwargs = self._generation_kwargs(max_tokens, temperature, top_p)
        streamer = TextIteratorStreamer(self.tokenizer,
                                        skip_prompt=True,
                                        skip_special_tokens=True)
        thread = Thread(target=self.model.generate,
                        kwargs={**inputs, **kwargs, "streamer": streamer})
        thread.start()

        emitted = ""
        sent = 0
        for chunk in streamer:
            emitted += chunk
            trimmed, finished = self._trim_stop(emitted, stop)
            new_text = trimmed[sent:]
            if new_text:
                yield new_text
                sent = len(trimmed)
            if finished:
                break


def create_app(engine: LLMEngine) -> FastAPI:
    app = FastAPI(title="OpenAI-Compatible Server")
    app.add_middleware(CORSMiddleware,
                       allow_origins=["*"],
                       allow_methods=["*"],
                       allow_headers=["*"])

    def chat_response(text: str, prompt_tokens: int,
                      completion_tokens: int) -> dict:
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": engine.model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    def completion_response(text: str, prompt_tokens: int,
                             completion_tokens: int) -> dict:
        return {
            "id": f"cmpl-{uuid.uuid4().hex}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": engine.model_id,
            "choices": [{
                "index": 0,
                "text": text,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    def enforce_model(request_model: Optional[str]):
        if request_model and request_model != engine.model_id:
            raise HTTPException(status_code=400,
                                detail="指定されたmodelは現在ロード済みのモデルと一致しません")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest):
        enforce_model(payload.model)
        prompt = engine._apply_chat_template(payload.messages)
        if payload.stream:
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            def event_stream():
                total = ""
                for delta in engine.stream(prompt,
                                           payload.max_tokens,
                                           payload.temperature,
                                           payload.top_p,
                                           payload.stop):
                    total += delta
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": engine.model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": delta
                            },
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                prompt_tokens = engine._token_count(prompt)
                completion_tokens = engine._token_count(total)
                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": engine.model_id,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(),
                                      media_type="text/event-stream")

        result = engine.generate(prompt,
                                 payload.max_tokens,
                                 payload.temperature,
                                 payload.top_p,
                                 payload.stop)
        return JSONResponse(chat_response(result["text"],
                                          result["prompt_tokens"],
                                          result["completion_tokens"]))

    @app.post("/v1/completions")
    async def completions(payload: CompletionRequest):
        enforce_model(payload.model)
        prompt = payload.prompt
        if payload.stream:
            completion_id = f"cmpl-{uuid.uuid4().hex}"
            created = int(time.time())

            def event_stream():
                total = ""
                for delta in engine.stream(prompt,
                                           payload.max_tokens,
                                           payload.temperature,
                                           payload.top_p,
                                           payload.stop):
                    total += delta
                    chunk = {
                        "id": completion_id,
                        "object": "text_completion.chunk",
                        "created": created,
                        "model": engine.model_id,
                        "choices": [{
                            "index": 0,
                            "text": delta,
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                prompt_tokens = engine._token_count(prompt)
                completion_tokens = engine._token_count(total)
                final_chunk = {
                    "id": completion_id,
                    "object": "text_completion.chunk",
                    "created": created,
                    "model": engine.model_id,
                    "choices": [{
                        "index": 0,
                        "text": "",
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                }
                yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(),
                                      media_type="text/event-stream")

        result = engine.generate(prompt,
                                 payload.max_tokens,
                                 payload.temperature,
                                 payload.top_p,
                                 payload.stop)
        return JSONResponse(completion_response(result["text"],
                                                result["prompt_tokens"],
                                                result["completion_tokens"]))

    return app


def main():
    args = parse_args()
    cache_dir = Path(args.model_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    # 8bitが指定され、かつユーザの明示が無い場合は12GB級GPU向けにCPUオフロードを既定で有効化
    enable_offload = args.enable_cpu_offload or (args.quantization == "8bit")

    offload_dir = Path(args.offload_dir) if args.offload_dir else None
    engine = LLMEngine(
        args.model,
        args.quantization,
        cache_dir,
        enable_cpu_offload=enable_offload,
        gpu_mem_limit=args.gpu_mem_limit,
        cpu_mem_limit=args.cpu_mem_limit,
        offload_dir=offload_dir,
    )
    app = create_app(engine)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

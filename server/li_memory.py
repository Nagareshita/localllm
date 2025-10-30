from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import os


def _ensure_embedding_available(repo_id: str, local_dir: Path, offline: bool = False) -> Path:
    """Download embedding model into models dir if missing.

    - repo_id: e.g. "BAAI/bge-m3"
    - local_dir: e.g. Path("models/bge-m3")
    Returns local_dir.
    """
    # If directory already contains model weights/config, assume ready
    if local_dir.exists() and any(local_dir.glob("**/*")):
        return local_dir

    if offline:
        raise FileNotFoundError(f"[OFFLINE] Embedding model not found: {local_dir}")

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub が必要です。pip install huggingface_hub"
        ) from e

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        # sentence-transformers系/bge系の典型ファイル群
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.py",
            "*.safetensors",
            "*.bin",
            "tokenizer*",
            "config*",
            "model*",
        ],
        ignore_patterns=["*.onnx", "*.gguf", "*.md"],
    )
    return local_dir


@dataclass
class LIMemoryConfig:
    # 注意: メモリの要約に使うLLMはサーバ内の"no memory"エンドポイントを叩く
    api_base: str = "http://localhost:8003/v1-nomem"
    api_key: str = "dummy"
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    short_turn_threshold: int = 10          # 10ターンまではChatMemoryBuffer重視
    summary_token_limit: int = 4000         # 要約用メモリの目安トークン
    chat_token_limit: int = 1500            # 直近チャットメモリの目安トークン
    vector_top_k: int = 5                   # ベクタ検索
    vector_every_n_turns: int = 3           # 何ターンごとにVectorMemoryへ蓄積するか
    embed_repo_id: str = "BAAI/bge-m3"
    embed_local_dir: str = "models/bge-m3"
    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "li_memory_default"


class LIMemoryManager:
    """
    - 直近 <= Nターン: ChatMemoryBuffer
    - 超過: ChatSummaryMemoryBuffer（要約・圧縮）
    - 定期: VectorMemory（後からの検索参照）
    - 統合: ComposableMemory で get(query) からメモリ文脈を取得
    """

    def __init__(self, cfg: Optional[LIMemoryConfig] = None):
        self.cfg = cfg or LIMemoryConfig()

        # 依存の読み込み（遅延インポート + エラーメッセージ最小）
        try:
            from llama_index.core.memory import (
                ChatMemoryBuffer,
                ChatSummaryMemoryBuffer,
                VectorMemory,
            )
            try:
                from llama_index.core.memory import (
                    SimpleComposableMemory as ComposableMemory,
                )
            except Exception:
                from llama_index.core.memory import ComposableMemory  # type: ignore

            from llama_index.core.schema import Document
            from llama_index.llms.openai_like import OpenAILike
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            # Qdrant vector store
            try:
                from llama_index.vector_stores.qdrant import QdrantVectorStore
            except Exception:
                from llama_index.vector_stores import QdrantVectorStore  # type: ignore
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except Exception as e:
            raise RuntimeError(
                "LlamaIndex関連のモジュールが見つかりません。"
                "pip install llama-index llama-index-llms-openai-like llama-index-embeddings-huggingface"
            ) from e

        # 要約用LLM（同一FastAPIの no-memory エンドポイントを使用して再帰注入を回避）
        self.llm = OpenAILike(
            api_base=self.cfg.api_base,
            api_key=self.cfg.api_key,
            model=self.cfg.model,
            temperature=0.3,      # 要約用は低温度
            max_tokens=512,
            is_chat_model=True,
            timeout=60,
        )

        # 埋め込みモデルのローカル確認/取得（bge-m3）
        offline = os.environ.get("TRANSFORMERS_OFFLINE", "") == "1"
        local_dir = _ensure_embedding_available(
            self.cfg.embed_repo_id, Path(self.cfg.embed_local_dir), offline=offline
        )

        # 埋め込み
        self.embed_model = HuggingFaceEmbedding(
            model_name=str(local_dir),
            cache_folder=str(local_dir),
        )

        # Qdrant クライアントとコレクション（存在しなければ作成）
        # Docker等の外部Qdrantが無い場合、埋め込みモードをサポート
        use_embedded = os.environ.get("QDRANT_EMBEDDED", "") == "1" or not self.cfg.qdrant_url or self.cfg.qdrant_url == "embedded"
        if use_embedded:
            data_dir = Path.cwd() / "server" / "qdrant_data"
            data_dir.mkdir(parents=True, exist_ok=True)
            self._qdrant_client = QdrantClient(path=str(data_dir))
        else:
            self._qdrant_client = QdrantClient(url=self.cfg.qdrant_url, api_key=self.cfg.qdrant_api_key)
        # 埋め込み次元を推定してコレクション作成
        try:
            probe = self.embed_model.get_text_embedding("dimension probe")
            dim = len(probe)
        except Exception:
            dim = 1024  # bge-m3 の既定次元（フォールバック）
        try:
            self._qdrant_client.get_collection(self.cfg.qdrant_collection)
        except Exception:
            self._qdrant_client.recreate_collection(
                collection_name=self.cfg.qdrant_collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

        self.vector_store = QdrantVectorStore(
            collection_name=self.cfg.qdrant_collection,
            client=self._qdrant_client,
        )

        # メモリ群
        self.chat_mem = ChatMemoryBuffer.from_defaults(
            token_limit=self.cfg.chat_token_limit
        )
        self.summary_mem = ChatSummaryMemoryBuffer.from_defaults(
            llm=self.llm, token_limit=self.cfg.summary_token_limit
        )
        self.vector_mem = VectorMemory.from_defaults(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
            retriever_kwargs={"similarity_top_k": self.cfg.vector_top_k},
        )
        # まとめ
        self.memory = ComposableMemory.from_defaults(
            memories=[self.summary_mem, self.chat_mem, self.vector_mem]
        )
        self.turn_count = 0
        self._Document = Document

    def build_context(self, query: str) -> str:
        """ComposableMemoryから問い合わせに適したメモリ文脈を取り出す。"""
        try:
            ctx = self.memory.get(query_str=query)
        except TypeError:
            ctx = self.memory.get(query)
        return ctx or ""

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        """対話1ターンをメモリに反映。"""
        self.turn_count += 1

        # 短期: 直近の会話をそのまま入れる
        self.chat_mem.put(f"User: {user_text}")
        self.chat_mem.put(f"Assistant: {assistant_text}")

        # 長期: しきい値を超えたらSummaryMemory側にも投入（内部で要約・圧縮）
        if self.turn_count > self.cfg.short_turn_threshold:
            self.summary_mem.put(
                f"User: {user_text}\nAssistant: {assistant_text}"
            )

        # 定期ベクタ格納
        if self.turn_count % self.cfg.vector_every_n_turns == 0:
            chunk = f"{user_text}\n{assistant_text}"
            self.vector_mem.put(self._Document(text=chunk))

    def reset(self) -> None:
        self.chat_mem.reset()
        self.summary_mem.reset()
        # VectorMemoryは用途に応じて消す/残すを選べる
        # （現状は保持）
        self.turn_count = 0

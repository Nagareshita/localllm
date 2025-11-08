"""
Qdrant Chat History Manager
チャット履歴をQdrantに保存・取得するバックエンドサービス
"""
import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, VectorParams, PointStruct, Filter,
                                  FieldCondition, MatchValue)
import uvicorn


# Pydanticモデル
class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class SaveHistoryRequest(BaseModel):
    session_id: str
    messages: List[Message]


class GetHistoryRequest(BaseModel):
    session_id: str


# FastAPIアプリ
app = FastAPI(title="Chat History API")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Qdrantクライアント初期化
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")
COLLECTION_NAME = "chat_history"

client = QdrantClient(path=QDRANT_PATH)

# コレクション初期化
try:
    client.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' already exists.")
except Exception:
    # ダミーベクトル用（履歴管理のみなので1次元）
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1, distance=Distance.COSINE)
    )
    print(f"Created collection '{COLLECTION_NAME}'.")


@app.post("/save_history")
async def save_history(request: SaveHistoryRequest):
    """チャット履歴を保存"""
    try:
        point_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # メッセージにタイムスタンプを追加
        messages_with_timestamp = []
        for msg in request.messages:
            msg_dict = msg.model_dump()
            if not msg_dict.get("timestamp"):
                msg_dict["timestamp"] = timestamp
            messages_with_timestamp.append(msg_dict)
        
        # Qdrantに保存
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=[0.0],  # ダミーベクトル
                    payload={
                        "session_id": request.session_id,
                        "messages": messages_with_timestamp,
                        "timestamp": timestamp,
                        "message_count": len(messages_with_timestamp)
                    }
                )
            ]
        )
        
        return {
            "status": "success",
            "point_id": point_id,
            "message_count": len(messages_with_timestamp)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_history")
async def get_history(request: GetHistoryRequest):
    """セッションIDから履歴を取得"""
    try:
        # セッションIDで検索（Filterオブジェクトを使用）
        qfilter = Filter(must=[
            FieldCondition(key="session_id",
                           match=MatchValue(value=request.session_id))
        ])
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qfilter,
            limit=100,
            with_payload=True,
        )

        if not points:
            return {"session_id": request.session_id, "messages": []}

        # 最新の履歴を取得
        latest_point = max(points,
                           key=lambda x: (x.payload or {}).get("timestamp",
                                         ""))

        return {
            "session_id": request.session_id,
            "messages": latest_point.payload.get("messages", []),
            "timestamp": latest_point.payload.get("timestamp")
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear_history")
async def clear_history(request: GetHistoryRequest):
    """セッションIDの履歴を削除"""
    try:
        # セッションIDで検索して削除（Filterオブジェクトを使用し、ページング）
        qfilter = Filter(must=[
            FieldCondition(key="session_id",
                           match=MatchValue(value=request.session_id))
        ])

        point_ids: list[str] = []
        offset = None
        while True:
            points, next_page = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=qfilter,
                limit=256,
                with_payload=False,
                offset=offset,
            )
            point_ids.extend([p.id for p in points])
            if not next_page:
                break
            offset = next_page
        
        if point_ids:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=point_ids
            )
        
        return {
            "status": "success",
            "deleted_count": len(point_ids)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list_sessions")
async def list_sessions(limit: int = 1000):
    """保存されているセッションID一覧を返す（重複除外・昇順）"""
    try:
        sessions = set()
        offset = None
        while True:
            points, next_page = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=None,
                limit=min(256, max(1, limit - len(sessions))),
                with_payload=True,
                offset=offset,
            )
            for p in points:
                sid = (p.payload or {}).get("session_id")
                if sid:
                    sessions.add(sid)
                if len(sessions) >= limit:
                    break
            if not next_page or len(sessions) >= limit:
                break
            offset = next_page
        return {"sessions": sorted(sessions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "healthy", "qdrant_path": QDRANT_PATH}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8004,
        log_level="info"
    )

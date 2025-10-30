# セットアップガイド（WSLサーバ + Windowsクライアント）

このプロジェクトは「WSL側でAPIサーバ」「Windows側でHTMLクライアント」を分離しています。VectorMemoryはWSL側でQdrantに永続化します。

## ディレクトリ構成

- `server/`（WSLで起動）
  - `api.py` FastAPIアプリ（/v1/chat/completions, /api/sessions, /health など）
  - `runner.py` モデルのロードと生成
  - `li_memory.py` メモリ統合（Chat/Summary/Vector + Qdrant）
  - セッションは `sessions/*.json` に保存
- `client/`（Windowsでブラウザから開く）
  - `qwen_chat.html` シンプルなUI（新規/既存チャット切替に対応）
（互換レイヤは廃止しました）

## 1) 環境作成（Python 3.10）

```bash
mamba create -n rag python=3.10 -y
conda activate rag
```

## 2) PyTorch/cu121 を固定

```bash
mamba install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

## 3) pip / uv / uvx の準備

```bash
python -m pip install --upgrade pip
# uv を入れて高速インストール（推奨）
python -m pip install uv
# もしくは pipx を使う場合
# pipx install uv
```

補足: `uvx` は uv に同梱のランチャです（`uvx <ツール>` で一時実行）。

## 4) 依存のインストール

```bash
uv pip install -r requirements.txt
```

## 5) Qdrant の起動（VectorMemoryの永続化）

選択肢A: Docker（推奨・簡単）
```bash
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

選択肢B: Dockerなし（埋め込みモード／WSLでそのまま実行）
- 追加インストールは不要です。サーバ起動前に以下の環境変数をセットしてください。
```bash
export QDRANT_EMBEDDED=1
```
- これで `server/li_memory.py` が `server/qdrant_data/` 配下に埋め込みQdrantを自動生成し、永続化します。
- 外部Qdrantを使いたくなったら `QDRANT_EMBEDDED` を外し、`server/li_memory.py` の `LIMemoryConfig.qdrant_url` を `http://localhost:6333` に戻してください（既定）。

## 6) APIサーバの起動（WSL）

```bash
export QWEN_CFG_PATH=$PWD/server/config.json   # 設定ファイル
uvicorn server.api:app --host 0.0.0.0 --port 8003

# またはシェルスクリプト
chmod +x server/start.sh
server/start.sh
```

エンドポイント:
- ヘルスチェック: `GET http://localhost:8003/health`
- OpenAI互換: `POST http://localhost:8003/v1/chat/completions`
- 内部用（メモリ無効）: `POST http://localhost:8003/v1-nomem/chat/completions`
- セッション管理: `GET/POST http://localhost:8003/api/sessions`, `POST /api/sessions/{sid}/chat`

## 7) クライアント（Windows）

`client/qwen_chat.html` をブラウザで開きます。既定で `http://localhost:8003` のAPIに接続します。

## 8) 埋め込みモデル（bge-m3）

初回アクセス時、`models/bge-m3` が無ければ自動ダウンロードします。オフライン運用時は事前に配置してください。

## 9) 補足

- OpenAI互換は非ストリームの Chat Completions のみ（ストリーミング/ツールは未対応）
- LlamaIndexが未インストールでもAPIは動作（メモリ機能は無効化）
- Qdrantのコレクションは自動作成（bge-m3に合わせた次元）。名前は `li_memory_default`（`LIMemoryConfig`で変更可）

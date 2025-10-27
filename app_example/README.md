アプリ例（APIクライアント）

このフォルダには、`qwen_app.py` の API を呼び出す超簡単なクライアント例を3つ用意しています。

前提
- API サーバーを起動しておく
  - PowerShell:
    - `conda activate rag`
    - `$env:QWEN_CFG_PATH = "$PWD\config.json"`
    - `uvicorn qwen_app:app --host 127.0.0.1 --port 8000`

1) CLI 例: `api_chat_cli.py`
- 対話: `python app_example/api_chat_cli.py`
- 1回だけ: `python app_example/api_chat_cli.py --once "PythonでFizzBuzz"`
- エンドポイント変更: `--url http://127.0.0.1:8000/generate`

2) Tkinter 例: `api_chat_gui_tk.py`
- 実行: `python app_example/api_chat_gui_tk.py`
- 送信で API に投げて結果が表示されます
- エンドポイント: 環境変数 `QWEN_API_URL` またはファイル内 `URL` を編集

3) ブラウザ例: `api_chat_web.html`
- 実行: ダブルクリックで開く（ファイルをそのまま開けます）
- 送信で API に POST します（`http://127.0.0.1:8000/generate`）
- 備考: サーバー側に CORS 設定を追加済み（開発用途、ワイルドカード許可）

注意
- 先に API サーバーを起動しておかないと、各クライアントはエラーになります
- Windows では conda 環境を有効化して `python` で実行してください（`py` は非推奨）


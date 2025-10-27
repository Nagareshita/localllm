WSLでAPIサーバーを動かし、Windowsから使うための超やさしい手順書

この説明書だけ見れば、PCに詳しくない方でも「WSL上でAPIサーバーを起動し、Windowsから会話できる」状態まで動かせます。

できること
- WSL（Ubuntu）の中で重いモデルを実行（GPU/量子化 も選べる）
- Windows側のツール（コマンド、簡易GUI、ブラウザ）から呼び出して使う

準備（最初の一回だけ）
- Windows 10/11 で WSL2 と Ubuntu を入れておく
  - 不明な場合は「Windowsの検索」→「PowerShell」を管理者で開き、次を実行: `wsl --install -d Ubuntu`
  - PCを再起動し、Ubuntu の初期設定（ユーザ名とパスワード）を済ませます
- このリポジトリを Windows 側に置きます（例: `C:\git\RAG_system\ModiGen\Local_LLM`）

手順A. VS Codeのターミナル（PowerShell）からWSLを開く
1) VS Code を開く（フォルダ `C:\git\RAG_system\ModiGen\Local_LLM` を開いておくと楽です）
2) メニュー「ターミナル」→「新しいターミナル」を開く（PowerShell）
3) 次を入力してWSL（Ubuntu）へ入ります
   - `wsl`
   - プロンプトが `$` になればOK（Ubuntuの世界です）
4) プロジェクトフォルダに移動します
   - `cd /mnt/c/git/RAG_system/ModiGen/Local_LLM`

手順B. WSLの中に専用の環境（llmapi）を用意する
0) もしConda/Mambaが無い場合は「Miniforge」で検索し、インストールしてください（公式の手順でOK）。インストール後に新しいWSLターミナルを開きます。
1) 環境を作る（Python 3.10 固定）
   - `mamba create -n llmapi python=3.10 -y`
2) 環境を有効化
   - `conda activate llmapi`

手順C. 必要なソフトを入れる（GPUあり/なしで分岐）
- 先に「uv（uvx 含む）」を入れます（高速で安全なパッケージ管理ツール）
  - インストール（WSL/Ubuntu）:
    - `curl -LsSf https://astral.sh/uv/install.sh | sh`
    - パスを通す（入っていない場合）: `export PATH="$HOME/.local/bin:$PATH"`
    - 確認: `uv --version` と `uvx --version`

GPU（NVIDIA）を使う場合（おすすめ）
- NVIDIAドライバはWindows側に入っている前提です
- 次を実行（CUDA 12.1 の例）
  - `mamba install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y`

CPUのみで使う場合（動きますが遅いです）
- `uv pip install --system torch==2.3.1`

次に、アプリ本体の必要パッケージを入れます（以降は pip の代わりに uv を使います）
- `uv pip install --system -r requirements.txt`

量子化（メモリ節約のオプション）も使う場合
- 4bit/8bit 量子化を使うには、WSLで次を実行
  - `uv pip install --system bitsandbytes`

参考: requirements.txt の内容（この通り入ります）
```
# ===== Core RAG Framework =====
langchain==0.3.27
langchain-openai==0.3.32
langchain-community==0.3.27
langgraph==0.2.38

# ===== LLM Providers =====
azure-identity==1.17.1
azure-search-documents==11.5.3
azure-storage-blob==12.23.0
azure-ai-vision-imageanalysis==1.0.0b3
azure-ai-documentintelligence==1.0.0b3
openai>=1.0.0
anthropic>=0.25.0
boto3>=1.34.0
requests>=2.31.0
httpx==0.27.2

# ===== Vector DB =====
qdrant-client==1.15.1

# ===== Document Processing =====
pymupdf==1.26.5
pymupdf4llm==0.0.27
python-docx==1.1.2
beautifulsoup4==4.12.3
datasets==2.19.1
pyarrow==15.0.2
Markdown==3.6
FlagEmbedding==1.2.10
openpyxl>=3.1.5

# ===== Tree-sitter =====
tree-sitter==0.20.4

# ===== Web Framework =====
fastapi==0.115.0
uvicorn==0.35.0

# ===== GUI Framework =====
pyside6==6.9.2
pyside6-addons==6.9.2
dearpygui==1.11.1

# ===== Evaluation =====
ragas==0.1.19

# ===== Utilities =====
python-dotenv==1.0.1
pydantic==2.9.2
redis==5.0.8

# ===== Dev =====
pytest==8.3.3
pytest-asyncio==0.23.8
black==24.8.0

# ===== Image / Numerics =====
opencv-python-headless==4.11.0.86
Pillow>=11.0.0
numpy>=1.26.0

# ===== Multi-agent Extras =====
jsonschema==4.23.0
aiofiles==23.2.0
psutil==6.0.0

# ===== VLM 実績ピン（SAIL-VL2 / Qwen2-VLで安定）=====
transformers==4.51.3
tokenizers==0.21.4
huggingface-hub==0.35.3
accelerate==1.10.1
safetensors==0.6.2
einops>=0.6.1
sentencepiece>=0.2.0
timm>=0.9.12,<1.0.0

# =====エクセル出力（pdfのキャプション）=====
openpyxl>=3.1.5

# =====MCP=====
mcp>=1.2.0
```

手順D. 量子化の設定（しなくてもOK）
- ファイル `config.json` を用意・編集します（このフォルダにあります）
- 量子化しない（最初はこれでOK）
```
{
  "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "torch_dtype": "float16",
  "device_map": "auto"
}
```
- 8bit量子化（メモリを節約したいときのおすすめ。精度はほぼ維持）
```
{
  "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "torch_dtype": "float16",
  "device_map": "auto",
  "quantization": "bnb-8bit"
}
```
- 4bit量子化（さらに軽いが、少し癖が出ることあり）
```
{
  "model_id": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "torch_dtype": "float16",
  "device_map": "auto",
  "quantization": "bnb-4bit",
  "bnb_4bit_compute_dtype": "bfloat16",
  "bnb_4bit_use_double_quant": true,
  "bnb_4bit_quant_type": "nf4"
}
```
メモ
- 量子化を使うときは、WSLで `pip install bitsandbytes` を忘れずに。
- モデルファイルは自動で `models/` に保存されます（自分で置く必要はありません）。

手順E. APIサーバーをWSLで起動する
1) まだなら `conda activate llmapi`
2) 次を実行
```
export QWEN_CFG_PATH="$PWD/config.json"
uvicorn qwen_app:app --host 127.0.0.1 --port 8000
```
3) 初回はモデルを自動ダウンロードします（少し時間がかかります）
4) 2回目からはローカルの `models/` を使います。完全オフラインで使うなら以下も設定
```
export TRANSFORMERS_OFFLINE=1
```

手順F. Windowsから使ってみる
- CLIクライアント（会話）
  - PowerShellで: `python app_example\api_chat_cli.py --url http://127.0.0.1:8000/generate`
- 簡易GUI（Tkinter）
  - PowerShellで: `python app_example\api_chat_gui_tk.py`
- ブラウザ
  - `app_example\api_chat_web.html` をダブルクリックで開く

チェック（困ったとき）
- GPUが使えているか（WSLで）
  - `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
  - True ならOK。False のときはGPU用のPyTorchやWSL2/NVIDIAドライバを確認
- 遅い/応答が来ない
  - 最初の1回はやや遅いのが普通。その後は速くなります
  - `config.json` の `max_new_tokens` を 128〜256 に下げると体感が軽くなります
  - 8bit/4bit量子化を検討
- Windowsからアクセスできない
  - ほとんどの環境では `127.0.0.1` で届きます
  - だめな場合は、WSLで `uvicorn ... --host 0.0.0.0` とし、`http://<WSLのIP>:8000` を使います（`ip addr show eth0` でIP確認）

よくある質問
- WSLに何を置けばいい？
  - Windows側に置いたこのフォルダをWSLから開いて使えます（`/mnt/c/...` 経由）。特別にWSLにコピーする必要はありません
- 量子化は必須？
  - 必須ではありません。まずは量子化なし（float16）で試し、重い/VRAMが足りない場合に 8bit→4bit の順で検討してください
- APIが起動していれば、Windowsのどのツールからも使える？
  - はい。付属のCLI/Tkinter/ブラウザ例に加え、他のプログラムからHTTPで呼べます

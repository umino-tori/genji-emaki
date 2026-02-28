# 源氏物語 AI絵巻 — Genji RAG

源氏物語のテキストをベクトル化し、RAG（Retrieval Augmented Generation）で質問に根拠付きで答える Streamlit チャットアプリです。

## 概要

青空文庫の源氏物語テキストを Google の埋め込みモデルでベクトル化して ChromaDB に保存し、ユーザーの質問に対して関連する場面を検索したうえで Gemini LLM が「平安文学の専門家」として回答します。

## スクリーンショット

| ライトモード（平安風） | ダークモード（Windows風） |
|---|---|
| 和紙を模した暖色系テーマ | モダンなダークテーマ |

## 技術スタック

| 項目 | 技術・モデル |
|---|---|
| 埋め込みモデル | Google `gemini-embedding-001` |
| LLM（回答生成） | Google `gemini-2.5-flash-lite` |
| ベクトルDB | ChromaDB（PersistentClient） |
| Web UI | Streamlit |
| 言語 | Python 3.10+ |

ローカルLLMとして `Qwen/Qwen2.5-7B-Instruct-GGUF` も選択可能です（カスタムAPIエンドポイント経由）。

## ディレクトリ構成

```
.
├── src/
│   ├── app.py               # Streamlit RAGチャットアプリ
│   └── genji_vector_db.py   # ベクトルDB構築・検索・評価 CLIツール
├── data/
│   ├── raw/                 # 入力JSONデータ（源氏物語 5帖分）
│   └── chroma_db/           # ChromaDB 永続化データ
├── requirements.txt
└── .env                     # Google APIキー設定（要作成）
```

## 収録データ

`data/raw/` に収録されている源氏物語（全56帖中 5帖）:

| ファイル | 帖名 | 読み仮名 |
|---|---|---|
| 源氏物語_json作成_01kiritsubo.json | 桐壺 | きりつぼ |
| 源氏物語_json作成_02hahakigi.json | 帚木 | ははきぎ |
| 源氏物語_json作成_03utsusemi.json | 空蝉 | うつせみ |
| 源氏物語_json作成_04yugao.json | 夕顔 | ゆうがお |
| 源氏物語_json作成_05wakamurasaki.json | 若紫 | わかむらさき |

各JSONチャンクには原文・主語補完済みの検索用テキスト・登場人物・タグ・場所などのメタデータが含まれます。

## セットアップ

### 前提条件

- Python 3.10 以上
- [Google AI Studio](https://aistudio.google.com/) で発行した API キー

### インストール

```bash
# 1. リポジトリをクローン
git clone https://github.com/<your-username>/genji-emaki.git
cd genji-emaki

# 2. 仮想環境の作成と有効化
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. 依存ライブラリのインストール
pip install -r requirements.txt

# 4. .env ファイルに API キーを設定
```echo 'GOOGLE_API_KEY="your-api-key-here"' > .env
```
下記にて使用  
gemini-embedding-001
gemini-2.5-flash-lite

😊カスタムLLMにつて
なお、ほかにローカルLLMとしてGoogle Colab上で動作するQwen/Qwen2.5-7B-Instruct-GGUFを使用することも可能です。
ローカルLLMの設定は、src/app.pyの「CUSTOM_API_BASE_URL」を設定してください。
「CUSTOM_API_BASE_URL」は、https://github.com/umino-tori/genji-emaki_google-colab/server_local_llm.ipynbで取得した「🚀 URL:https://xxxxxxxxxxxxxxxx..ngrok-free.app」を設定してください。

### ベクトルDBの構築

```bash
python src/genji_vector_db.py ingest
```

`data/raw/` のJSONを読み込み、`gemini-embedding-001` でベクトル化して ChromaDB に保存します。

### アプリの起動

```bash
streamlit run src/app.py
```

ブラウザで `http://localhost:8501` にアクセスしてください。

## CLIツール（genji_vector_db.py）

### データ投入

```bash
python src/genji_vector_db.py ingest
```

### 対話型検索

```bash
# 対話モード
python src/genji_vector_db.py search

# クエリを直接指定
python src/genji_vector_db.py search --query "光源氏と藤壺の密通"

# 取得件数を変更（デフォルト: 5件）
python src/genji_vector_db.py search --query "夕顔の死" --n-results 3
```

### 検索精度の評価

```bash
python src/genji_vector_db.py eval
```

12件のテストケースで **Recall@k** を計測します。

| クエリ例 | 期待キーワード |
|---|---|
| 光源氏と藤壺の密通 | 藤壺 |
| 帝が最も愛した女性の死 | 桐壺 |
| 夕顔の死と怪異 | 夕顔 |
| 若紫との出会い | 若紫 |

## RAG処理フロー

```
ユーザーの質問
    │
    ▼
gemini-embedding-001 でベクトル化
    │
    ▼
ChromaDB でコサイン類似度検索（上位 k 件）
    │
    ▼
原文 + 解説テキストをコンテキストとして構築
    │
    ▼
Gemini LLM（平安文学の専門家ペルソナ）が回答生成
    │
    ▼
ストリーミング表示 ＋ 参照した場面をカード表示
```

## 機能一覧

- **チャットUI**: `st.chat_message` によるモダンなチャット形式
- **テーマ切り替え**: 平安風ライトモード / Windows風ダークモード
- **現代語訳付き引用**: 原文引用に現代語訳を自動付与するオプション
- **検索結果カード**: 参照した場面を帖名・関連度・タグ・登場人物付きで表示
- **収録帖一覧**: サイドバーに読み仮名付きで帖一覧を表示
- **LLM切り替え**: Gemini / ローカルQwen の選択
- **ログ記録**: 質問・検索時間・ヒット情報を `app.py.log` に記録

## 依存ライブラリ

```
google-generativeai
chromadb
tqdm
python-dotenv
streamlit
```

## ライセンス

テキストデータは[青空文庫](https://www.aozora.gr.jp/)の源氏物語（与謝野晶子訳）を使用しています。

"""
源氏物語 AI絵巻 - Genji RAG -
ChromaDB を利用した源氏物語の RAG 検索チャットアプリケーション
"""

import os
import time
import logging

import requests
import streamlit as st
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv

# .env からAPIキーを読み込み
load_dotenv()

# --- 定数 ---
EMBEDDING_MODEL = "models/gemini-embedding-001"
# LLM_MODEL = "models/gemini-2.5-flash-preview-09-2025"
LLM_MODEL = "models/gemini-2.5-flash-lite"
DB_PATH = "./data/chroma_db"
COLLECTION_NAME = "genji_story"
LOG_FILE = "app.py.log"
CUSTOM_API_BASE_URL = "https://5ea6-34-125-70-68.ngrok-free.app"

# --- 帖名 → 読み仮名 対応表（全54帖＋番外） ---
CHAPTER_READINGS = {
    "桐壺":   "きりつぼ",
    "帚木":   "ははきぎ",
    "空蝉":   "うつせみ",
    "夕顔":   "ゆうがお",
    "若紫":   "わかむらさき",
    "末摘花": "すえつむはな",
    "紅葉賀": "もみじのが",
    "花宴":   "はなのえん",
    "葵":     "あおい",
    "賢木":   "さかき",
    "花散里": "はなちるさと",
    "須磨":   "すま",
    "明石":   "あかし",
    "澪標":   "みおつくし",
    "蓬生":   "よもぎう",
    "関屋":   "せきや",
    "絵合":   "えあわせ",
    "松風":   "まつかぜ",
    "薄雲":   "うすぐも",
    "朝顔":   "あさがお",
    "乙女":   "おとめ",
    "玉鬘":   "たまかずら",
    "初音":   "はつね",
    "胡蝶":   "こちょう",
    "蛍":     "ほたる",
    "常夏":   "とこなつ",
    "篝火":   "かがりび",
    "野分":   "のわき",
    "行幸":   "みゆき",
    "藤袴":   "ふじばかま",
    "真木柱": "まきばしら",
    "梅枝":   "うめがえ",
    "藤裏葉": "ふじのうらは",
    "若菜上": "わかなじょう",
    "若菜下": "わかなげ",
    "柏木":   "かしわぎ",
    "横笛":   "よこぶえ",
    "鈴虫":   "すずむし",
    "夕霧":   "ゆうぎり",
    "御法":   "みのり",
    "幻":     "まぼろし",
    "雲隠":   "くもがくれ",
    "匂宮":   "におうみや",
    "紅梅":   "こうばい",
    "竹河":   "たけかわ",
    "橋姫":   "はしひめ",
    "椎が本": "しいがもと",
    "総角":   "あげまき",
    "早蕨":   "さわらび",
    "宿木":   "やどりぎ",
    "東屋":   "あずまや",
    "浮舟":   "うきふね",
    "蜻蛉":   "かげろう",
    "手習":   "てならい",
    "夢浮橋": "ゆめのうきはし",
}

SYSTEM_PROMPT = (
    "あなたは源氏物語に精通した平安文学の専門家です。"
    "優雅で丁寧な現代日本語（です・ます調）で、質問に答えてください。"
    "古語や古文の内容は、現代語に噛み砕いてわかりやすく解説してください。"
    "現代では使いにくい漢字や難読漢字には、括弧書きで読み仮名を付与してください"
    "（例：更衣（こうい）、帚木（ははきぎ）、御息所（みやすどころ））。"
    "回答の際は、必ず検索された『根拠テキスト』の内容に基づき、"
    "どの帖のどの場面かも補足してください。"
)

MODERN_TRANSLATION_INSTRUCTION = (
    "回答の中で根拠テキストの原文を引用する際は、"
    "必ず以下のフォーマットで原文と現代語訳をセットで提示してください。\n"
    "フォーマット:\n"
    "**【原文】**\n> （原文の引用）\n\n"
    "**【現代語訳】**\n> （上記原文のわかりやすい現代語訳）\n\n"
    "原文を引用するたびに、必ずこのフォーマットを使ってください。"
)

# --- ログ設定 ---
file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

app_logger = logging.getLogger("genji_rag_app")
app_logger.setLevel(logging.INFO)
# 重複ハンドラ防止
if not app_logger.handlers:
    app_logger.addHandler(file_handler)


# --- キャッシュ付きリソース ---
@st.cache_resource
def get_chroma_collection():
    """ChromaDB コレクションを取得する（キャッシュ）"""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def search_chroma(query, api_key, n_results=4):
    """クエリでChromaDBを検索し、結果を返す"""
    genai.configure(api_key=api_key)

    # クエリをベクトル化
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="RETRIEVAL_QUERY",
    )
    query_embedding = result["embedding"]

    # ChromaDB検索
    collection = get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return results


def build_context(results):
    """検索結果からLLM用のコンテキスト文字列を組み立てる"""
    context_parts = []
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0])
    ):
        chapter = meta.get("chapter", "")
        tags = meta.get("tags", "")
        content_to_embed = meta.get("content_to_embed", "")
        context_parts.append(
            f"【根拠テキスト {i + 1}】帖: {chapter} / タグ: {tags}\n"
            f"原文: {doc[:500]}\n"
            f"解説: {content_to_embed[:500]}"
        )
    return "\n\n".join(context_parts)


def _build_prompt(query, context, with_modern_translation=False):
    """LLM用プロンプトを構築する"""
    extra = ""
    if with_modern_translation:
        extra = f"\n--- 出力フォーマット指示 ---\n{MODERN_TRANSLATION_INSTRUCTION}\n"
    return (
        f"以下の検索された根拠テキストに基づいて、ユーザーの質問に答えてください。\n\n"
        f"--- 検索された根拠テキスト ---\n{context}\n"
        f"{extra}"
        f"--- 質問 ---\n{query}"
    )


def generate_answer_stream(query, context, api_key, with_modern_translation=False):
    """Gemini LLM でストリーミング回答を生成する"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=LLM_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )
    prompt = _build_prompt(query, context, with_modern_translation)

    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text


def generate_answer(query, context, api_key, with_modern_translation=False):
    """Gemini LLM で回答を生成する（非ストリーミング、フォールバック用）"""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=LLM_MODEL,
        system_instruction=SYSTEM_PROMPT,
    )
    prompt = _build_prompt(query, context, with_modern_translation)

    response = model.generate_content(prompt)
    return response.text


def generate_answer_custom_api(query, context, with_modern_translation=False):
    """カスタムAPI で回答を生成する"""
    prompt = _build_prompt(query, context, with_modern_translation)
    full_system_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    res = requests.post(
        f"{CUSTOM_API_BASE_URL}/llms",
        json={
            "system_prompt": full_system_prompt,
            "max_tokens": 6000,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        timeout=60,
    )
    res.raise_for_status()
    return res.json()["response"]


def cosine_distance_to_similarity(distance):
    """コサイン距離を類似度パーセントに変換する"""
    similarity = max(0.0, 1.0 - distance)
    return similarity * 100


def parse_csv_metadata(value):
    """カンマ区切りのメタデータ文字列をリストに戻す"""
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def render_search_results_styled(results, tag_bg, tag_text, char_bg, char_text):
    """検索結果をテーマに合わせたスタイルで表示する"""
    st.markdown("---")
    st.markdown("#### 参照した物語の場面")

    for i, (doc_id, doc, meta, dist) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        chapter = meta.get("chapter", "")
        similarity = cosine_distance_to_similarity(dist)
        tags = parse_csv_metadata(meta.get("tags", ""))
        characters = parse_csv_metadata(meta.get("characters", ""))
        location = meta.get("location", "")

        with st.expander(
            f"**{chapter}** ── 関連度: {similarity:.0f}%",
            expanded=(i == 0),
        ):
            # メタデータ行
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**帖名:** {chapter}")
                if location:
                    st.markdown(f"**場所:** {location}")
                st.caption(f"チャンクID: {doc_id}")
            with col2:
                if tags:
                    tag_html = " ".join(
                        f'<span style="background-color:{tag_bg};color:{tag_text};'
                        f'padding:2px 8px;border-radius:12px;margin:2px;'
                        f'font-size:0.85em;display:inline-block;">{t}</span>'
                        for t in tags
                    )
                    st.markdown(f"**タグ:** {tag_html}", unsafe_allow_html=True)
                if characters:
                    char_html = " ".join(
                        f'<span style="background-color:{char_bg};color:{char_text};'
                        f'padding:2px 8px;border-radius:12px;margin:2px;'
                        f'font-size:0.85em;display:inline-block;">{c}</span>'
                        for c in characters
                    )
                    st.markdown(
                        f"**登場人物:** {char_html}", unsafe_allow_html=True
                    )

            # 類似度バー
            st.progress(similarity / 100, text=f"関連度: {similarity:.1f}%")

            # 原文引用
            st.markdown("**原文:**")
            excerpt = doc[:300]
            st.markdown(f"> {excerpt}...")


# --- メイン画面 ---
def main():
    st.set_page_config(
        page_title="源氏物語 AI絵巻",
        page_icon="🌸",
        layout="wide",
    )

    # --- APIキーを .env から取得 ---
    api_key = os.getenv("GOOGLE_API_KEY", "")

    # --- サイドバー ---
    with st.sidebar:
        st.subheader("モデル選択")
        model_type = st.radio(
            "使用するLLM",
            ["Gemini", "ローカルQwen/Qwen2.5-7B-Instruct-GGUF"],
            index=0,
            horizontal=True,
            help="回答生成に使用するLLMを切り替えます",
        )

        st.header("設定")

        st.subheader("テーマ設定")
        theme = st.radio(
            "表示モード",
            ["ライト（平安風）", "ダーク（Windows風）"],
            index=0,
            help="画面のデザインを切り替えます"
        )

        # テーマに応じたCSS変数の定義（表示前に必要）
        if theme == "ライト（平安風）":
            bg_color = "#faf6f0"
            text_color = "#3d2b1f"
            sidebar_bg = "#f3ece2"
            sidebar_text = "#3d2b1f"
            title_color = "#6b3a5c"
            card_bg = "#ffffff"
            card_border = "#e0d4c8"
            caption_color = "#7a6a5e"
            quote_color = "#4a3728"
            tag_bg = "#e8d5b7"
            tag_text = "#5c3d2e"
            char_bg = "#d4e6d4"
            char_text = "#2d5a2d"
        else:  # ダーク（Windows風）
            bg_color = "#1a1a1a"
            text_color = "#e0e0e0"
            sidebar_bg = "#2d2d2d"
            sidebar_text = "#e0e0e0"
            title_color = "#d4a373"
            card_bg = "#333333"
            card_border = "#444444"
            caption_color = "#aaaaaa"
            quote_color = "#bbbbbb"
            tag_bg = "#5c4d37"
            tag_text = "#e8d5b7"
            char_bg = "#2d4a2d"
            char_text = "#d4e6d4"

        st.subheader("検索設定")
        n_results = st.slider(
            "取得件数 (k)",
            min_value=1,
            max_value=10,
            value=4,
            help="検索で取得するチャンク数",
        )

        st.subheader("表示設定")
        with_modern_translation = st.checkbox(
            "原文に現代語訳を付与",
            value=False,
            help="有効にすると、回答中の原文引用に現代語訳を併記します",
        )

        # DB接続状態とモデル情報の表示
        st.markdown("---")
        try:
            collection = get_chroma_collection()
            doc_count = collection.count()
            st.success(f"DB接続済み（{doc_count} 件のドキュメント）")

            # 帖名ごとのチャンク数を集計
            all_meta = collection.get(include=["metadatas"])["metadatas"]
            chapter_map = {}  # {chapter_name: {"count": int, "order": int}}
            for meta in all_meta:
                ch = meta.get("chapter", "不明")
                order = meta.get("chapter_order", 9999)
                if ch not in chapter_map:
                    chapter_map[ch] = {"count": 0, "order": order}
                chapter_map[ch]["count"] += 1

            sorted_chapters = sorted(chapter_map.items(), key=lambda x: x[1]["order"])

            # 帖一覧をカード風デザインで表示
            rows_html = "".join(
                f'<tr>'
                f'<td style="padding:3px 4px;color:{sidebar_text};">'
                f'<ruby>{ch}<rt style="font-size:0.8em;color:{sidebar_text};">'
                f'{CHAPTER_READINGS.get(ch, "")}'
                f'</rt></ruby>'
                f'</td>'
                f'<td style="padding:3px 4px;text-align:right;color:{sidebar_text};'
                f'font-family:monospace;font-size:0.85em;">{info["count"]} chunk</td>'
                f'</tr>'
                for ch, info in sorted_chapters
            )
            st.markdown(
                f'<div style="background-color:{card_bg};border:1px solid {card_border};'
                f'border-radius:8px;padding:12px;margin-top:10px;">'
                f'<div style="font-size:0.95em;font-weight:bold;margin-bottom:8px;'
                f'color:{sidebar_text};">収録帖一覧</div>'
                f'<table style="width:100%;border-collapse:collapse;font-size:0.9em;">'
                f'{rows_html}'
                f'</table>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # 使用中のモデル情報を表示
            llm_display = LLM_MODEL.split("/")[-1] if model_type == "Gemini" else "ローカルQwen/Qwen2.5-7B-Instruct-GGUF"
            st.markdown("---")
            st.markdown(
                f"""
                <div style="background-color: {card_bg}; border: 1px solid {card_border}; border-radius: 8px; padding: 12px; margin-top: 10px;">
                    <div style="font-size: 0.95em; font-weight: bold; margin-bottom: 4px;">ベクトルDB:</div>
                    <div style="font-size: 1.0em; font-family: monospace; margin-bottom: 12px;">ChromaDB</div>
                    <div style="font-size: 0.95em; font-weight: bold; margin-bottom: 4px;">ベクトルモデル:</div>
                    <div style="font-size: 1.0em; font-family: monospace; margin-bottom: 12px;">{EMBEDDING_MODEL.split('/')[-1]}</div>
                    <div style="font-size: 0.95em; font-weight: bold; margin-bottom: 4px;">LLMモデル:</div>
                    <div style="font-size: 1.0em; font-family: monospace;">{llm_display}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"DB接続エラー: {e}")


    # カスタムCSSの適用
    st.markdown(
        f"""
        <style>
        /* 全体 */
        .stApp {{
            background-color: {bg_color};
            color: {text_color};
        }}
        /* サイドバー */
        section[data-testid="stSidebar"] {{
            background-color: {sidebar_bg};
            color: {sidebar_text};
        }}
        section[data-testid="stSidebar"] * {{
            color: {sidebar_text} !important;
        }}
        section[data-testid="stSidebar"] .stSlider > div > div > div > div {{
            color: {sidebar_text} !important;
        }}
        /* メインコンテンツ */
        .stMainBlockContainer * {{
            color: {text_color};
        }}
        /* チャット */
        .stChatMessage {{
            border-radius: 12px;
        }}
        .stChatMessage * {{
            color: {text_color} !important;
        }}
        /* タイトル */
        h1 {{
            color: {title_color} !important;
            text-align: center;
        }}
        /* 検索結果カード */
        .stExpander {{
            background-color: {card_bg};
            border: 1px solid {card_border};
            border-radius: 10px;
        }}
        /* キャプション */
        .stCaption, caption {{
            color: {caption_color} !important;
        }}
        /* 引用ブロック */
        blockquote, blockquote * {{
            color: {quote_color} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("源氏物語 AI絵巻 ─ Genji RAG ─")
    st.caption("平安の識者AIが、源氏物語の世界をご案内いたします")

    # APIキー未設定時の案内（Gemini選択時のみ）
    if model_type == "Gemini" and not api_key:
        st.warning(
            ".env ファイルに GOOGLE_API_KEY が設定されていません。"
            "設定後、アプリを再起動してください。"
        )

    # --- チャット履歴の管理 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # チャット履歴を表示
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # 検索結果がある場合はカードも再表示
            if msg.get("search_results"):
                render_search_results_styled(msg["search_results"], tag_bg, tag_text, char_bg, char_text)

    # --- ユーザー入力 ---
    if prompt := st.chat_input("源氏物語について質問してください"):
        if model_type == "Gemini" and not api_key:
            st.error(".env ファイルに GOOGLE_API_KEY を設定してください。")
            return

        # ユーザーメッセージを表示・保存
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- RAG処理 ---
        with st.chat_message("assistant"):
            try:
                # 1. ベクトル検索
                search_start = time.time()
                results = search_chroma(prompt, api_key, n_results=n_results)
                search_time = time.time() - search_start

                # 2. コンテキスト構築
                context = build_context(results)

                # 3. LLM回答を表示（モデル選択に応じて切り替え）
                if model_type == "ローカルQwen/Qwen2.5-7B-Instruct-GGUF":
                    st.caption("🤖 モデル: ローカルQwen/Qwen2.5-7B-Instruct-GGUF")
                    answer = generate_answer_custom_api(
                        prompt, context, with_modern_translation
                    )
                    st.markdown(answer)
                else:
                    st.caption(f"🤖 モデル: {LLM_MODEL.split('/')[-1]}")
                    # Gemini: ストリーミング→非ストリーミングにフォールバック
                    try:
                        answer = st.write_stream(
                            generate_answer_stream(
                                prompt, context, api_key, with_modern_translation
                            )
                        )
                    except Exception:
                        answer = generate_answer(
                            prompt, context, api_key, with_modern_translation
                        )
                        st.markdown(answer)

                # 4. 検索結果カードを表示
                render_search_results_styled(results, tag_bg, tag_text, char_bg, char_text)

                # 5. 回答をセッションに保存
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "search_results": results,
                    }
                )

                # 6. ログ出力
                hit_info = [
                    f"{rid}({dist:.4f})"
                    for rid, dist in zip(
                        results["ids"][0], results["distances"][0]
                    )
                ]
                app_logger.info(
                    f"質問: {prompt} | "
                    f"検索時間: {search_time:.2f}s | "
                    f"ヒット: {', '.join(hit_info)} | "
                    f"回答抜粋: {answer[:100]}..."
                )

            except Exception as e:
                error_msg = f"エラーが発生しました: {e}"
                st.error(error_msg)
                app_logger.error(f"質問: {prompt} | エラー: {e}")


if __name__ == "__main__":
    main()

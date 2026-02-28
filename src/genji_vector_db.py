"""
源氏物語 ベクトルDB構築・検索・評価スクリプト
JSONデータをベクトル化してChromaDBに保存し、検索精度を検証する。
"""

import os
import sys
import json
import time
import glob
import logging
import argparse

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# 定数
EMBEDDING_MODEL = "models/gemini-embedding-001"
DATA_DIR = "data/raw"
DB_PATH = "./data/chroma_db"
COLLECTION_NAME = "genji_story"
EMBEDDING_BATCH_SIZE = 20  # 一度にAPIに送るテキスト数
API_SLEEP_SEC = 1.0  # API呼び出し間のスリープ（秒）

# 統計評価用テストケース
TEST_CASES = [
    {"query": "光源氏と藤壺の密通", "expected_keyword": "藤壺"},
    {"query": "帝が最も愛した女性の死", "expected_keyword": "桐壺"},
    {"query": "夕顔の死と怪異", "expected_keyword": "夕顔"},
    {"query": "若紫との出会い", "expected_keyword": "若紫"},
    {"query": "空蝉との逢瀬", "expected_keyword": "空蝉"},
    {"query": "帚木の巻の雨夜の品定め", "expected_keyword": "帚木"},
    {"query": "嫉妬に苦しむ後宮の女性たち", "expected_keyword": "桐壺"},
    {"query": "光源氏の美しい少女への執着", "expected_keyword": "若紫"},
    {"query": "もののけに襲われる女性", "expected_keyword": "夕顔"},
    {"query": "中流階級の女性についての議論", "expected_keyword": "帚木"},
    {"query": "身を隠す女性との恋", "expected_keyword": "空蝉"},
    {"query": "桐壺更衣の死と帝の悲しみ", "expected_keyword": "桐壺"},
]


class GenjiVectorDB:
    """源氏物語ベクトルDBクラス"""

    def __init__(self, db_path=DB_PATH, collection_name=COLLECTION_NAME):
        """初期化: APIキー読み込み、ChromaDB接続"""
        # .envファイルからAPIキーを読み込み
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY が .env に設定されていません。")
            sys.exit(1)

        genai.configure(api_key=api_key)
        logger.info("Google AI API の設定が完了しました。")

        # ChromaDB クライアント初期化
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB コレクション '{collection_name}' を取得/作成しました。"
            f"（既存ドキュメント数: {self.collection.count()}）"
        )

    def _load_json_files(self):
        """data/raw/ 配下のJSONファイルを全て読み込む"""
        pattern = os.path.join(DATA_DIR, "*.json")
        files = sorted(glob.glob(pattern))
        if not files:
            logger.error(f"JSONファイルが見つかりません: {pattern}")
            sys.exit(1)

        all_chunks = []
        for filepath in files:
            logger.info(f"読み込み中: {os.path.basename(filepath)}")
            with open(filepath, "r", encoding="utf-8") as f:
                chunks = json.load(f, strict=False)
            all_chunks.extend(chunks)

        logger.info(f"合計 {len(all_chunks)} チャンクを読み込みました。")
        return all_chunks

    def _embed_texts(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        """テキストリストをバッチでベクトル化する"""
        all_embeddings = []
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[i : i + EMBEDDING_BATCH_SIZE]
            try:
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=batch,
                    task_type=task_type,
                )
                all_embeddings.extend(result["embedding"])
            except Exception as e:
                logger.error(f"埋め込みAPIエラー (バッチ {i}): {e}")
                # リトライ: 少し待ってから再試行
                logger.info("10秒後にリトライします...")
                time.sleep(10)
                try:
                    result = genai.embed_content(
                        model=EMBEDDING_MODEL,
                        content=batch,
                        task_type=task_type,
                    )
                    all_embeddings.extend(result["embedding"])
                except Exception as e2:
                    logger.error(f"リトライも失敗しました: {e2}")
                    raise
            time.sleep(API_SLEEP_SEC)
        return all_embeddings

    def _prepare_metadata(self, chunk):
        """チャンクのメタデータをChromaDB用に変換する"""
        meta = chunk.get("metadata", {})
        result = {
            "chapter": chunk.get("chapter", ""),
            "chapter_order": chunk.get("chapter_order", 0),
            "content_to_embed": chunk.get("content_to_embed", ""),
        }

        # リスト型フィールドはカンマ区切り文字列に変換
        characters = meta.get("characters", [])
        if isinstance(characters, list):
            result["characters"] = ", ".join(characters)
        else:
            result["characters"] = str(characters) if characters else ""

        tags = meta.get("tags", [])
        if isinstance(tags, list):
            result["tags"] = ", ".join(tags)
        else:
            result["tags"] = str(tags) if tags else ""

        # その他のメタデータ（None は空文字に変換）
        result["location"] = meta.get("location") or ""
        result["scent"] = meta.get("scent") or ""
        result["color_costume"] = meta.get("color_costume") or ""

        return result

    def ingest_data(self):
        """JSONデータをベクトル化してChromaDBに保存する"""
        # 既にデータがある場合の確認
        existing_count = self.collection.count()
        if existing_count > 0:
            logger.warning(
                f"コレクションに既に {existing_count} 件のデータがあります。"
            )
            answer = input("既存データを削除して再作成しますか？ (y/N): ").strip().lower()
            if answer == "y":
                self.client.delete_collection(COLLECTION_NAME)
                self.collection = self.client.get_or_create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("コレクションを再作成しました。")
            else:
                logger.info("既存データを維持します。処理を中断します。")
                return

        # JSONデータ読み込み
        chunks = self._load_json_files()

        # ベクトル化するテキストを収集
        texts_to_embed = [chunk["content_to_embed"] for chunk in chunks]
        logger.info(f"{len(texts_to_embed)} 件のテキストをベクトル化します...")

        # ベクトル化
        embeddings = self._embed_texts(texts_to_embed, task_type="RETRIEVAL_DOCUMENT")
        logger.info("ベクトル化が完了しました。")

        # ChromaDBに保存
        logger.info("ChromaDB にデータを保存中...")
        for i, chunk in enumerate(tqdm(chunks, desc="ChromaDB保存")):
            self.collection.add(
                ids=[chunk["chunk_id"]],
                documents=[chunk["original_text"]],
                embeddings=[embeddings[i]],
                metadatas=[self._prepare_metadata(chunk)],
            )

        logger.info(
            f"保存完了: {self.collection.count()} 件のドキュメントが登録されました。"
        )

    def search(self, query, n_results=5):
        """クエリでベクトル検索を行い、結果を表示する"""
        if self.collection.count() == 0:
            logger.error(
                "コレクションにデータがありません。先に ingest を実行してください。"
            )
            return []

        # クエリをベクトル化
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="RETRIEVAL_QUERY",
        )
        query_embedding = result["embedding"]

        # ChromaDB検索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        # 結果を表示
        print(f"\n{'='*80}")
        print(f"検索クエリ: {query}")
        print(f"ヒット数: {len(results['ids'][0])}")
        print(f"{'='*80}")

        for i, (doc_id, doc, meta, dist) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            print(f"\n--- 結果 {i + 1} ---")
            print(f"  スコア（距離）: {dist:.4f}")
            print(f"  チャンクID   : {doc_id}")
            print(f"  帖名         : {meta.get('chapter', '')}")
            print(f"  タグ         : {meta.get('tags', '')}")
            print(f"  登場人物     : {meta.get('characters', '')}")
            print(f"  場所         : {meta.get('location', '')}")
            # 原文の抜粋（最初の150文字）
            excerpt = doc[:150].replace("\n", " ")
            print(f"  原文（抜粋） : {excerpt}...")
            # 検索用テキストの抜粋
            embed_text = meta.get("content_to_embed", "")
            embed_excerpt = embed_text[:150].replace("\n", " ")
            print(f"  検索用テキスト: {embed_excerpt}...")

        print(f"\n{'='*80}\n")
        return results

    def evaluate(self, k=5):
        """テストケースで検索精度を統計的に評価する"""
        if self.collection.count() == 0:
            logger.error(
                "コレクションにデータがありません。先に ingest を実行してください。"
            )
            return

        print(f"\n{'='*80}")
        print(f"統計的評価モード (Recall@{k})")
        print(f"テストケース数: {len(TEST_CASES)}")
        print(f"{'='*80}\n")

        hits = 0
        total = len(TEST_CASES)

        for tc in tqdm(TEST_CASES, desc="評価中"):
            query = tc["query"]
            expected = tc["expected_keyword"]

            # クエリをベクトル化
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=query,
                task_type="RETRIEVAL_QUERY",
            )
            query_embedding = result["embedding"]

            # 検索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["metadatas"],
            )

            # 上位k件に期待キーワードが含まれるか判定
            hit = False
            for meta in results["metadatas"][0]:
                chapter = meta.get("chapter", "")
                tags = meta.get("tags", "")
                characters = meta.get("characters", "")
                # 帖名、タグ、登場人物のいずれかに期待キーワードが含まれるか
                if expected in chapter or expected in tags or expected in characters:
                    hit = True
                    break

            status = "HIT" if hit else "MISS"
            print(f"  [{status}] クエリ: {query}")
            print(f"         期待: {expected}")
            if not hit:
                # ミスの場合、実際の帖名を表示
                found_chapters = [
                    m.get("chapter", "") for m in results["metadatas"][0]
                ]
                print(f"         実際: {found_chapters}")

            if hit:
                hits += 1

            time.sleep(API_SLEEP_SEC)

        # 結果サマリー
        recall = (hits / total) * 100 if total > 0 else 0
        print(f"\n{'='*80}")
        print(f"評価結果")
        print(f"  テストケース数 : {total}")
        print(f"  ヒット数       : {hits}")
        print(f"  Recall@{k}     : {recall:.1f}%")
        print(f"{'='*80}\n")


def interactive_search(db):
    """対話型検索モード"""
    print("\n源氏物語ベクトル検索（対話モード）")
    print("終了するには 'quit' または 'exit' と入力してください。\n")

    while True:
        query = input("検索クエリを入力: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("検索を終了します。")
            break
        db.search(query)


def main():
    parser = argparse.ArgumentParser(
        description="源氏物語ベクトルDB 構築・検索・評価ツール"
    )
    parser.add_argument(
        "mode",
        choices=["ingest", "search", "eval"],
        help="実行モード: ingest=データ投入, search=対話型検索, eval=統計評価",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="検索結果の表示件数 (デフォルト: 5)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="検索クエリ（searchモードで直接指定する場合）",
    )

    args = parser.parse_args()

    # GenjiVectorDB インスタンス生成
    db = GenjiVectorDB()

    if args.mode == "ingest":
        db.ingest_data()
    elif args.mode == "search":
        if args.query:
            db.search(args.query, n_results=args.n_results)
        else:
            interactive_search(db)
    elif args.mode == "eval":
        db.evaluate(k=args.n_results)


if __name__ == "__main__":
    main()

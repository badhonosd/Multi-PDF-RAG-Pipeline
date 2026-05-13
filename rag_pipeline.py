import os
import re
import json
import uuid
import hashlib
import requests
from typing import List, Dict, Any, Tuple

import pdfplumber
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter


# CONFIG


PDF_FOLDER = "input_pdfs"

PDF_FILES = sorted([
    os.path.join(PDF_FOLDER, file_name)
    for file_name in os.listdir(PDF_FOLDER)
    if file_name.lower().endswith(".pdf")
]) if os.path.exists(PDF_FOLDER) else []

CHROMA_DIR = "chroma_db_multi"
COLLECTION_NAME = "multi_pdf_rag_collection"

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_K_M"
OLLAMA_URL = "http://localhost:11434/api/generate"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

VECTOR_TOP_K = 8
BM25_TOP_K = 8
FINAL_TOP_K = 4

MIN_VECTOR_SCORE = 0.15
MIN_HYBRID_SCORE = 0.15

HASH_FILE = "pdf_hashes.json"




def normalize_text(text: str) -> str:
    """Clean text but preserve original case for display/context."""
    if not text:
        return ""
    text = text.replace("\u200c", " ").replace("\u200d", " ")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def normalize_for_search(text: str) -> str:
    """
    Normalize text for retrieval/search only.
    casefold() makes matching stronger than lower().
    """
    text = normalize_text(text)
    return text.casefold()


def safe_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenizer for BM25.
    Supports English + Bangla.
    """
    text = normalize_for_search(text)
    text = re.sub(r"[^\w\u0980-\u09FF]+", " ", text, flags=re.UNICODE)
    return text.split()


def cosine_from_distance(distance: float) -> float:
    """
    Convert Chroma distance into similarity-like score.
    """
    return 1.0 / (1.0 + float(distance))


def get_pdf_files() -> List[str]:
    if not os.path.exists(PDF_FOLDER):
        return []
    return sorted([
        os.path.join(PDF_FOLDER, file_name)
        for file_name in os.listdir(PDF_FOLDER)
        if file_name.lower().endswith(".pdf")
    ])


def file_md5(file_path: str) -> str:
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def load_pdf_hashes() -> Dict[str, str]:
    if not os.path.exists(HASH_FILE):
        return {}
    try:
        with open(HASH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_pdf_hashes(hashes: Dict[str, str]):
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        json.dump(hashes, f, ensure_ascii=False, indent=2)



# PDF EXTRACTION


def extract_pdf_content(pdf_path: str) -> List[Dict[str, Any]]:
    records = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            # -------- Page text --------
            page_text = normalize_text(page.extract_text() or "")

            if page_text:
                records.append({
                    "id": f"{os.path.basename(pdf_path)}-text-page-{page_index}",
                    "type": "page_text",
                    "page": page_index,
                    "content": page_text,
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "page": page_index,
                        "content_type": "text",
                        "section": "page_text"
                    }
                })

            # -------- Tables --------
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []

            for table_index, table in enumerate(tables, start=1):
                if not table:
                    continue

                cleaned_rows = []
                for row in table:
                    cleaned_row = [normalize_text(safe_str(cell)) for cell in row]
                    if any(cell for cell in cleaned_row):
                        cleaned_rows.append(cleaned_row)

                if not cleaned_rows:
                    continue

                max_cols = max(len(row) for row in cleaned_rows)
                padded_rows = [row + [""] * (max_cols - len(row)) for row in cleaned_rows]

                header = padded_rows[0]
                body_rows = padded_rows[1:] if len(padded_rows) > 1 else []

                # Full table as readable text
                table_lines = []
                table_lines.append(" | ".join(header))
                for row in body_rows:
                    table_lines.append(" | ".join(row))

                table_text = normalize_text("\n".join(table_lines))

                records.append({
                    "id": f"{os.path.basename(pdf_path)}-table-{page_index}-{table_index}",
                    "type": "table",
                    "page": page_index,
                    "content": table_text,
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "page": page_index,
                        "content_type": "table",
                        "table_index": table_index,
                        "header": json.dumps(header, ensure_ascii=False)
                    }
                })

                # Save each row separately for more precise retrieval
                for row_index, row in enumerate(body_rows, start=1):
                    row_pairs = []
                    for col_idx, cell in enumerate(row):
                        col_name = (
                            header[col_idx]
                            if col_idx < len(header) and header[col_idx]
                            else f"column_{col_idx + 1}"
                        )
                        row_pairs.append(f"{col_name}: {cell}")

                    row_text = normalize_text(
                        f"Source: {os.path.basename(pdf_path)} ; "
                        f"Page: {page_index} ; "
                        f"Table: {table_index} ; "
                        + " ; ".join(row_pairs)
                    )

                    if row_text:
                        records.append({
                            "id": f"{os.path.basename(pdf_path)}-table-row-{page_index}-{table_index}-{row_index}",
                            "type": "table_row",
                            "page": page_index,
                            "content": row_text,
                            "metadata": {
                                "source": os.path.basename(pdf_path),
                                "page": page_index,
                                "content_type": "table_row",
                                "table_index": table_index,
                                "row_index": row_index,
                                "header": json.dumps(header, ensure_ascii=False)
                            }
                        })

    return records



# CHUNKING


def build_chunks(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "।", ".", ";", " | ", " "]
    )

    chunks = []

    for record in records:
        content = normalize_text(record["content"])
        if not content:
            continue

        # Keep table/table_row intact
        if record["type"] in {"table", "table_row"}:
            chunk_texts = [content]
        else:
            chunk_texts = splitter.split_text(content)

        for idx, chunk_text in enumerate(chunk_texts, start=1):
            metadata = dict(record["metadata"])
            metadata["chunk_index"] = idx
            metadata["record_type"] = record["type"]

            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_text,                               # original text for context/LLM
                "search_text": normalize_for_search(chunk_text),  # normalized text for retrieval
                "metadata": metadata
            })

    return chunks



# RAG ENGINE


class MultiPDFRAG:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

        self.chroma_client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        self.all_chunks: List[Dict[str, Any]] = []
        self.bm25 = None
        self.bm25_corpus_tokens: List[List[str]] = []

    def _stringify_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                out[k] = "" if v is None else v
            else:
                out[k] = json.dumps(v, ensure_ascii=False)
        return out

    def index_pdfs(self, pdf_files: List[str]):
        all_records = []

        for pdf_path in pdf_files:
            if not os.path.exists(pdf_path):
                print(f"Skipped missing file: {pdf_path}")
                continue

            print(f"Extracting: {pdf_path}")
            records = extract_pdf_content(pdf_path)
            all_records.extend(records)

        if not all_records:
            raise ValueError("No extractable content found in the provided PDFs.")

        print("Building chunks...")
        chunks = build_chunks(all_records)

        if not chunks:
            raise ValueError("No chunks created from PDFs.")

        print(f"Total chunks: {len(chunks)}")

        # Clear old collection before reindexing
        existing = self.collection.count()
        if existing > 0:
            existing_data = self.collection.get(include=[])
            ids = existing_data.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)

        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            searchable_texts = [c["search_text"] for c in batch]   # embedding text
            original_texts = [c["text"] for c in batch]            # stored for context
            ids = [c["id"] for c in batch]
            metadatas = [self._stringify_metadata(c["metadata"]) for c in batch]

            embeddings = self.embedder.encode(
                searchable_texts,
                normalize_embeddings=True
            ).tolist()

            self.collection.add(
                ids=ids,
                documents=original_texts,
                metadatas=metadatas,
                embeddings=embeddings
            )

        self.all_chunks = chunks
        self.bm25_corpus_tokens = [tokenize_for_bm25(c["search_text"]) for c in chunks]
        self.bm25 = BM25Okapi(self.bm25_corpus_tokens)

        print("Indexing complete.")

    def add_pdfs_to_index(self, pdf_files: List[str]):
        all_records = []

        for pdf_path in pdf_files:
            if not os.path.exists(pdf_path):
                print(f"Skipped missing file: {pdf_path}")
                continue

            print(f"Extracting new/updated PDF: {pdf_path}")
            records = extract_pdf_content(pdf_path)
            all_records.extend(records)

        if not all_records:
            return

        print("Building chunks for new/updated PDFs...")
        chunks = build_chunks(all_records)

        if not chunks:
            return

        print(f"New/updated chunks: {len(chunks)}")

        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            searchable_texts = [c["search_text"] for c in batch]
            original_texts = [c["text"] for c in batch]
            ids = [c["id"] for c in batch]
            metadatas = [self._stringify_metadata(c["metadata"]) for c in batch]

            embeddings = self.embedder.encode(
                searchable_texts,
                normalize_embeddings=True
            ).tolist()

            self.collection.add(
                ids=ids,
                documents=original_texts,
                metadatas=metadatas,
                embeddings=embeddings
            )

        self.all_chunks.extend(chunks)
        self.bm25_corpus_tokens = [tokenize_for_bm25(c["search_text"]) for c in self.all_chunks]
        self.bm25 = BM25Okapi(self.bm25_corpus_tokens)

        print("New PDFs added to index.")

    def delete_source_from_index(self, source_name: str):
        try:
            data = self.collection.get(include=["metadatas"])
            ids = data.get("ids", [])
            metas = data.get("metadatas", [])

            delete_ids = []
            for chunk_id, meta in zip(ids, metas):
                if str(meta.get("source", "")) == source_name:
                    delete_ids.append(chunk_id)

            if delete_ids:
                self.collection.delete(ids=delete_ids)
                print(f"Deleted old indexed chunks for: {source_name}")

            self.all_chunks = [
                c for c in self.all_chunks
                if str(c["metadata"].get("source", "")) != source_name
            ]

            self.bm25_corpus_tokens = [
                tokenize_for_bm25(c["search_text"]) for c in self.all_chunks
            ]
            self.bm25 = BM25Okapi(self.bm25_corpus_tokens) if self.bm25_corpus_tokens else None

        except Exception as e:
            print(f"Failed to delete source {source_name}: {e}")

    def load_existing_index(self):
        data = self.collection.get(include=["documents", "metadatas"])
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])

        self.all_chunks = []
        for chunk_id, doc, meta in zip(ids, docs, metas):
            doc = normalize_text(doc)
            self.all_chunks.append({
                "id": chunk_id,
                "text": doc,
                "search_text": normalize_for_search(doc),
                "metadata": dict(meta)
            })

        if not self.all_chunks:
            raise ValueError("No indexed data found.")

        self.bm25_corpus_tokens = [
            tokenize_for_bm25(c["search_text"]) for c in self.all_chunks
        ]
        self.bm25 = BM25Okapi(self.bm25_corpus_tokens)

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        query = normalize_text(query)
        if not query:
            return []

        search_query = normalize_for_search(query)

        # -------- Vector Search --------
        query_embedding = self.embedder.encode(
            [search_query],
            normalize_embeddings=True
        ).tolist()[0]

        vector_result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=VECTOR_TOP_K,
            include=["documents", "metadatas", "distances"]
        )

        vector_docs = vector_result.get("documents", [[]])[0]
        vector_metas = vector_result.get("metadatas", [[]])[0]
        vector_distances = vector_result.get("distances", [[]])[0]

        vector_hits = []
        for doc, meta, dist in zip(vector_docs, vector_metas, vector_distances):
            sim = cosine_from_distance(dist)
            vector_hits.append({
                "text": doc,
                "metadata": dict(meta),
                "vector_score": sim
            })

        # -------- BM25 Search --------
        query_tokens = tokenize_for_bm25(search_query)
        bm25_scores = self.bm25.get_scores(query_tokens) if self.bm25 else []

        bm25_ranked_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:BM25_TOP_K]

        max_bm25 = max([bm25_scores[i] for i in bm25_ranked_indices], default=0.0)

        bm25_hits = []
        for idx in bm25_ranked_indices:
            score = bm25_scores[idx]
            normalized_score = (score / max_bm25) if max_bm25 > 0 else 0.0
            bm25_hits.append({
                "text": self.all_chunks[idx]["text"],
                "metadata": self.all_chunks[idx]["metadata"],
                "bm25_score": normalized_score
            })

        # -------- Merge Results --------
        merged = {}

        def key_for(item: Dict[str, Any]) -> str:
            meta = item["metadata"]
            return (
                f"{meta.get('source', '')}-"
                f"{meta.get('page', '')}-"
                f"{meta.get('record_type', '')}-"
                f"{item['text'][:120]}"
            )

        for hit in vector_hits:
            k = key_for(hit)
            merged[k] = {
                "text": hit["text"],
                "metadata": hit["metadata"],
                "vector_score": hit.get("vector_score", 0.0),
                "bm25_score": 0.0
            }

        for hit in bm25_hits:
            k = key_for(hit)
            if k not in merged:
                merged[k] = {
                    "text": hit["text"],
                    "metadata": hit["metadata"],
                    "vector_score": 0.0,
                    "bm25_score": hit.get("bm25_score", 0.0)
                }
            else:
                merged[k]["bm25_score"] = hit.get("bm25_score", 0.0)

        final_hits = []
        for item in merged.values():
            hybrid_score = 0.80 * item["vector_score"] + 0.20 * item["bm25_score"]
            item["hybrid_score"] = hybrid_score
            final_hits.append(item)

        final_hits.sort(key=lambda x: x["hybrid_score"], reverse=True)

        filtered = [
            h for h in final_hits
            if h["vector_score"] >= MIN_VECTOR_SCORE or h["hybrid_score"] >= MIN_HYBRID_SCORE
        ]

        return filtered[:FINAL_TOP_K]

    def build_context(self, hits: List[Dict[str, Any]]) -> str:
        blocks = []

        for i, hit in enumerate(hits, start=1):
            meta = hit["metadata"]
            block = [
                f"[Context {i}]",
                f"Source: {meta.get('source', '')}",
                f"Page: {meta.get('page', '')}",
                f"ContentType: {meta.get('content_type', '')}",
                f"RecordType: {meta.get('record_type', '')}",
                "Content:",
                hit["text"]
            ]
            blocks.append("\n".join(block))

        return "\n\n".join(blocks)

    def ask_llm(self, query: str, context: str) -> str:
        prompt = f"""
You are a strict question-answering assistant.

Rules:
1. Answer only from the provided context.
2. If the answer is present in the context, give the exact answer as clearly as possible.
3. If the answer is not present in the context, write exactly: NOT FOUND
4. Do not make up any information.
5. Prefer short and precise answers.
6. If multiple sources contain the answer, mention source name and page.
7. If table data contains the answer, prioritize table data.

Question:
{query}

Context:
{context}

Final Answer:
""".strip()

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0
            }
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            answer = normalize_text(data.get("response", ""))
            if not answer:
                return "NOT FOUND"
            return answer
        except Exception as e:
            return f"LLM ERROR: {e}"

    def answer(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        hits = self.retrieve(query)

        if not hits:
            return "NOT FOUND", []

        context = self.build_context(hits)
        answer = self.ask_llm(query, context)

        if not answer or answer.strip().upper() == "NOT FOUND":
            return "NOT FOUND", hits

        if answer.strip().lower() in {"", "not found", "not available", "no information found"}:
            return "NOT FOUND", hits

        return answer, hits



# AUTO SYNC FOR NEW / UPDATED / REMOVED PDFS


def sync_pdfs(engine: MultiPDFRAG):
    current_files = get_pdf_files()
    current_hashes = {}

    for pdf_path in current_files:
        try:
            current_hashes[os.path.basename(pdf_path)] = file_md5(pdf_path)
        except Exception as e:
            print(f"Hash error for {pdf_path}: {e}")

    saved_hashes = load_pdf_hashes()

    new_or_updated = []
    removed_files = []

    current_names = set(current_hashes.keys())
    saved_names = set(saved_hashes.keys())

    for pdf_path in current_files:
        name = os.path.basename(pdf_path)
        if name not in saved_hashes or saved_hashes[name] != current_hashes[name]:
            new_or_updated.append(pdf_path)

    for old_name in saved_names - current_names:
        removed_files.append(old_name)

    for removed_name in removed_files:
        print(f"Removed PDF detected: {removed_name}")
        engine.delete_source_from_index(removed_name)

    for pdf_path in new_or_updated:
        source_name = os.path.basename(pdf_path)
        if source_name in saved_hashes:
            print(f"Updated PDF detected: {source_name}")
            engine.delete_source_from_index(source_name)
        else:
            print(f"New PDF detected: {source_name}")

    if new_or_updated:
        engine.add_pdfs_to_index(new_or_updated)

    save_pdf_hashes(current_hashes)



# OUTPUT


def print_sources(hits: List[Dict[str, Any]]):
    if not hits:
        return

    print("\n--- Retrieved Sources ---")
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        print(
            f"{i}. source={meta.get('source')} | "
            f"page={meta.get('page')} | "
            f"type={meta.get('content_type')} | "
            f"record={meta.get('record_type')} | "
            f"score={hit.get('hybrid_score', 0):.4f}"
        )


def save_answer(answer: str, hits: List[Dict[str, Any]], output_file="answer_output.txt"):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Answer:\n")
        f.write(answer + "\n\n")

        if hits:
            f.write("Sources:\n")
            for i, hit in enumerate(hits, start=1):
                meta = hit["metadata"]
                f.write(
                    f"{i}. Source={meta.get('source')} | "
                    f"Page={meta.get('page')} | "
                    f"Type={meta.get('content_type')} | "
                    f"Record={meta.get('record_type')} | "
                    f"Score={hit.get('hybrid_score', 0):.4f}\n"
                )



# MAIN


def main():
    engine = MultiPDFRAG()

    available_pdfs = get_pdf_files()
    if not available_pdfs:
        print("No PDF files found.")
        return

    needs_index = True
    try:
        if os.path.exists(CHROMA_DIR) and engine.collection.count() > 0:
            engine.load_existing_index()
            needs_index = False
            print("Loaded existing vector database.")
    except Exception:
        needs_index = True

    if needs_index:
        engine.index_pdfs(available_pdfs)
        current_hashes = {
            os.path.basename(pdf): file_md5(pdf)
            for pdf in available_pdfs
        }
        save_pdf_hashes(current_hashes)
    else:
        sync_pdfs(engine)

    print("\nMulti-PDF RAG system is ready.")
    print("Loaded PDFs:")
    for pdf in get_pdf_files():
        print("-", pdf)

    print("\nType your query. Type 'exit' to quit.\n")

    while True:
        # auto detect before every query prompt
        sync_pdfs(engine)

        query = input("Query: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        if not query:
            print("NOT FOUND\n")
            save_answer("NOT FOUND", [])
            continue

        # auto detect again after user enters query
        sync_pdfs(engine)

        answer, hits = engine.answer(query)

        print("\nAnswer:")
        print(answer)
        print_sources(hits)

        save_answer(answer, hits)


if __name__ == "__main__":
    main()

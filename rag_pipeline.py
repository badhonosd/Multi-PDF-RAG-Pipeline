"""
Multi-PDF RAG Pipeline
Supports both English and Bangla queries.
Models: BAAI/bge-m3 (embeddings), Qwen/Qwen2.5-7B-Instruct (generation)
"""

import glob
import os
import re
import gc
import json
import uuid
import hashlib
import pdfplumber
import chromadb
import torch
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

PDF_DIR = "./data/pdfs"          # Change to your PDF folder
PDF_FILES = sorted(glob.glob(os.path.join(PDF_DIR, "**/*.pdf"), recursive=True))

EMBED_MODEL_NAME = "BAAI/bge-m3"
GEN_MODEL_NAME   = "Qwen/Qwen2.5-7B-Instruct"

CHROMA_DIR       = "./chroma_db_multi"
COLLECTION_NAME  = "multi_pdf_rag_collection"
HASH_FILE        = "./pdf_hashes.json"

CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

EN_VECTOR_TOP_K     = 8
EN_BM25_TOP_K       = 8
EN_FINAL_TOP_K      = 3
BN_VECTOR_TOP_K     = 10
BN_FINAL_TOP_K      = 5
EN_MIN_VECTOR_SCORE = 0.10
EN_MIN_HYBRID_SCORE = 0.08

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def normalize_text(text):
    if not text:
        return ""
    text = str(text)
    text = text.replace("\u200c", " ").replace("\u200d", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

def normalize_for_search(text):
    return normalize_text(text).casefold()

def tokenize_text(text):
    return re.findall(r"[\w\u0980-\u09FF]+", normalize_for_search(text))

def is_bangla(text):
    return bool(re.search(r"[\u0980-\u09FF]", text))

def safe_str(value):
    return "" if value is None else str(value).strip()

def file_md5(file_path):
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()

def load_hashes():
    if not os.path.exists(HASH_FILE):
        return {}
    try:
        with open(HASH_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_hashes(hashes):
    with open(HASH_FILE, "w", encoding="utf-8") as f:
        json.dump(hashes, f, ensure_ascii=False, indent=2)

def cosine_from_distance(distance):
    return 1.0 / (1.0 + float(distance))


# ──────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL_NAME, device=DEVICE)

print("Loading generation model...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL_NAME, trust_remote_code=True)

if DEVICE == "cuda":
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True
    )

model.eval()


# ──────────────────────────────────────────────
# PDF EXTRACTION & CHUNKING
# ──────────────────────────────────────────────

def extract_pdf_content(pdf_path):
    records = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            txt = normalize_text(page.extract_text() or "")
            if txt:
                records.append({
                    "id": f"{os.path.basename(pdf_path)}-text-page-{page_no}",
                    "type": "page_text",
                    "page": page_no,
                    "content": txt,
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "page": page_no,
                        "content_type": "text",
                        "section": "page_text"
                    }
                })
    return records


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_chunks(records):
    chunks = []
    for record in records:
        content = normalize_text(record["content"])
        if not content:
            continue
        for idx, ch in enumerate(chunk_text(content), start=1):
            metadata = dict(record["metadata"])
            metadata["chunk_index"] = idx
            metadata["record_type"] = record["type"]
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": ch,
                "search_text": normalize_text(ch),
                "bm25_text": normalize_for_search(ch),
                "metadata": metadata
            })
    return chunks


# ──────────────────────────────────────────────
# MAIN RAG CLASS
# ──────────────────────────────────────────────

class MultiPDFRAG:
    def __init__(self):
        self.client     = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.chunks: list = []
        self.bm25         = None

    # ── indexing ──

    def clear_collection(self):
        try:
            old = self.collection.get()
            if old["ids"]:
                self.collection.delete(ids=old["ids"])
        except Exception:
            pass

    def index_all_pdfs(self):
        if not PDF_FILES:
            raise ValueError("No PDFs found in PDF_DIR.")

        print("Extracting and chunking PDFs...")
        all_records = []
        for pdf_path in PDF_FILES:
            all_records.extend(extract_pdf_content(pdf_path))

        self.chunks = build_chunks(all_records)
        if not self.chunks:
            raise ValueError("No chunks created.")

        print("Total chunks:", len(self.chunks))
        self.clear_collection()

        embed_texts    = [c["search_text"] for c in self.chunks]
        original_texts = [c["text"]        for c in self.chunks]

        print("Embedding chunks...")
        embeddings = embedder.encode(embed_texts, normalize_embeddings=True, show_progress_bar=True)

        self.collection.add(
            ids        = [c["id"]       for c in self.chunks],
            documents  = original_texts,
            embeddings = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
            metadatas  = [c["metadata"] for c in self.chunks]
        )

        print("Building BM25...")
        corpus    = [tokenize_text(c["bm25_text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus)

        save_hashes({os.path.basename(p): file_md5(p) for p in PDF_FILES})
        print("Setup Done ✅")

    def load_bm25_from_chunks(self):
        data = self.collection.get(include=["documents", "metadatas"])
        self.chunks = [
            {
                "id": str(uuid.uuid4()),
                "text": doc,
                "search_text": normalize_text(doc),
                "bm25_text": normalize_for_search(doc),
                "metadata": meta
            }
            for doc, meta in zip(data.get("documents", []), data.get("metadatas", []))
        ]
        corpus    = [tokenize_text(c["bm25_text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def bootstrap(self):
        current_hashes = {os.path.basename(p): file_md5(p) for p in PDF_FILES}
        needs_reindex  = True
        try:
            if self.collection.count() > 0 and current_hashes == load_hashes():
                self.load_bm25_from_chunks()
                needs_reindex = False
                print("Loaded existing index.")
        except Exception:
            pass
        if needs_reindex:
            print("Rebuilding index...")
            self.index_all_pdfs()

    def sync_if_needed(self):
        current_hashes = {os.path.basename(p): file_md5(p) for p in PDF_FILES}
        if current_hashes != load_hashes():
            print("PDF change detected. Reindexing...")
            self.index_all_pdfs()

    # ── retrieval ──

    def retrieve_english(self, query):
        raw_query  = normalize_text(query)
        norm_query = normalize_for_search(query)

        q_emb   = embedder.encode([raw_query], normalize_embeddings=True)
        vec_res = self.collection.query(
            query_embeddings = q_emb.tolist() if hasattr(q_emb, "tolist") else q_emb,
            n_results        = EN_VECTOR_TOP_K,
            include          = ["documents", "metadatas", "distances"]
        )

        vector_hits = [
            {"text": doc, "metadata": meta,
             "vector_score": cosine_from_distance(dist), "bm25_score": 0.0}
            for doc, meta, dist in zip(
                vec_res["documents"][0],
                vec_res["metadatas"][0],
                vec_res.get("distances", [[]])[0]
            )
        ]

        bm_scores = self.bm25.get_scores(tokenize_text(norm_query)) if self.bm25 else []
        top_idx   = sorted(range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True)[:EN_BM25_TOP_K]
        max_bm25  = max((bm_scores[i] for i in top_idx), default=0.0)

        bm25_hits = [
            {"text": self.chunks[i]["text"], "metadata": self.chunks[i]["metadata"],
             "vector_score": 0.0,
             "bm25_score": (bm_scores[i] / max_bm25) if max_bm25 > 0 else 0.0}
            for i in top_idx
        ]

        merged = {}
        def make_key(item):
            m = item["metadata"]
            return f"{m.get('source','')}-{m.get('page','')}-{item['text'][:160]}"

        for item in vector_hits + bm25_hits:
            k = make_key(item)
            if k not in merged:
                merged[k] = item
            else:
                merged[k]["vector_score"] = max(merged[k]["vector_score"], item["vector_score"])
                merged[k]["bm25_score"]   = max(merged[k]["bm25_score"],   item["bm25_score"])

        final_hits = sorted(
            [{**item, "hybrid_score": 0.85 * item["vector_score"] + 0.15 * item["bm25_score"]}
             for item in merged.values()],
            key=lambda x: x["hybrid_score"], reverse=True
        )

        filtered = [h for h in final_hits
                    if h["vector_score"] >= EN_MIN_VECTOR_SCORE or h["hybrid_score"] >= EN_MIN_HYBRID_SCORE]
        return (filtered or final_hits)[:EN_FINAL_TOP_K]

    def retrieve_bangla(self, query):
        raw_query = normalize_text(query)
        q_emb     = embedder.encode([raw_query], normalize_embeddings=True)
        vec_res   = self.collection.query(
            query_embeddings = q_emb.tolist() if hasattr(q_emb, "tolist") else q_emb,
            n_results        = BN_VECTOR_TOP_K,
            include          = ["documents", "metadatas", "distances"]
        )
        hits = [
            {"text": doc, "metadata": meta,
             "vector_score": cosine_from_distance(dist),
             "bm25_score": 0.0,
             "hybrid_score": cosine_from_distance(dist)}
            for doc, meta, dist in zip(
                vec_res["documents"][0],
                vec_res["metadatas"][0],
                vec_res.get("distances", [[]])[0]
            )
        ]
        hits.sort(key=lambda x: x["vector_score"], reverse=True)
        return hits[:BN_FINAL_TOP_K]

    def retrieve(self, query):
        query = normalize_text(query)
        if not query:
            return []
        return self.retrieve_bangla(query) if is_bangla(query) else self.retrieve_english(query)

    # ── generation ──

    def build_context(self, hits):
        blocks = []
        for i, hit in enumerate(hits, start=1):
            meta = hit["metadata"]
            blocks.append(
                f"[Context {i}]\n"
                f"Source: {meta.get('source', '')}\n"
                f"Page: {meta.get('page', '')}\n"
                f"ContentType: {meta.get('content_type', '')}\n"
                f"RecordType: {meta.get('record_type', '')}\n"
                f"Content:\n{hit['text']}\n"
            )
        return "\n\n".join(blocks)

    def ask(self, query):
        self.sync_if_needed()

        hits = self.retrieve(query)
        if not hits:
            return "NOT FOUND", []

        prompt = f"""Answer only from the provided context.

Rules:
- If the answer is found in the context, return ONLY the answer.
- If the exact answer is not found but relevant information is clearly present, return the closest correct answer from the context.
- If the answer is truly not found in the context, return ONLY: NOT FOUND
- Do NOT return both.
- Do NOT add explanation.
- Answer in the same language as the question.
- Do not use outside knowledge.

Question: {query}

Context:
{self.build_context(hits)}

Answer:""".strip()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        outputs = model.generate(
            **inputs,
            max_new_tokens = 60,
            do_sample      = False,
            use_cache      = True,
            pad_token_id   = tokenizer.eos_token_id
        )

        ans = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip().split("\n")[0].strip()

        if not ans:
            ans = "NOT FOUND"
        if ans.casefold() in {"not found", "not available", "no information found", "unknown"}:
            ans = "NOT FOUND"

        return ans, hits


# ──────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────

def print_sources(hits):
    if not hits:
        return
    print("\nRetrieved Sources:\n")
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        print(
            f"{i}. {meta.get('source')} | "
            f"page {meta.get('page')} | "
            f"type {meta.get('content_type')} | "
            f"score {hit.get('hybrid_score', hit.get('vector_score', 0)):.4f}"
        )


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    engine = MultiPDFRAG()
    engine.bootstrap()

    while True:
        query = input("\nQuery: ").strip()
        if query.casefold() in {"exit", "quit"}:
            print("Exiting...")
            break
        if not query:
            print("NOT FOUND")
            continue
        try:
            answer, hits = engine.ask(query)
            print("\nAnswer:\n")
            print(answer)
            print_sources(hits)
        except torch.cuda.OutOfMemoryError:
            print("GPU MEMORY FULL")
            gc.collect()
            torch.cuda.empty_cache()

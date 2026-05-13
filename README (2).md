# Multi-PDF RAG Pipeline

> 📓 This project was originally developed as a **Kaggle Notebook**.  
> Kaggle link: _<!-- তোমার Kaggle notebook-এর link এখানে দাও -->_

Bangla + English দুটো ভাষাতেই PDF থেকে প্রশ্নের উত্তর দেওয়ার জন্য একটি Retrieval-Augmented Generation (RAG) pipeline।

## Features

- ✅ Bangla ও English উভয় query সাপোর্ট
- ✅ Hybrid retrieval — Vector search (BGE-M3) + BM25
- ✅ 4-bit quantization (GPU-তে কম VRAM লাগে)
- ✅ Multiple PDF indexing with MD5-based change detection
- ✅ ChromaDB persistent vector store

## Models

| Role | Model |
|------|-------|
| Embeddings | `BAAI/bge-m3` |
| Generation | `Qwen/Qwen2.5-7B-Instruct` |

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. PDF ফাইল রাখো
mkdir -p data/pdfs
# তোমার PDF গুলো data/pdfs/ ফোল্ডারে দাও

# 4. Run
python rag_pipeline.py
```

## Usage

Script চালানোর পর:

```
Query: বাংলাদেশের রাজধানী কী?
Answer: ঢাকা

Query: What is the capital of Bangladesh?
Answer: Dhaka

Query: exit   ← বন্ধ করতে
```

## Project Structure

```
├── rag_pipeline.py     # Main pipeline
├── requirements.txt    # Dependencies
├── data/
│   └── pdfs/           # তোমার PDF ফাইলগুলো এখানে রাখো
├── chroma_db_multi/    # Auto-generated vector store (gitignore করো)
└── pdf_hashes.json     # Auto-generated hash file (gitignore করো)
```

## .gitignore Suggestion

```
chroma_db_multi/
pdf_hashes.json
data/
__pycache__/
*.pyc
.env
```

## Kaggle-এ Run করতে

Original notebook Kaggle-এ develop করা হয়েছে। Kaggle-এ চালাতে `rag_pipeline.py`-এর config section-এ এই পরিবর্তন করো:

```python
PDF_DIR    = "/kaggle/input"
CHROMA_DIR = "/kaggle/working/chroma_db_multi"
HASH_FILE  = "/kaggle/working/pdf_hashes.json"
```

এবং notebook-এর শুরুতে:
```python
!pip -q install pdfplumber chromadb rank_bm25 sentence-transformers transformers accelerate bitsandbytes
```

## Notes

- GPU না থাকলে CPU-তেও চলবে, তবে অনেক ধীর হবে।
- Local-এ চালাতে `data/pdfs/` ফোল্ডারে PDF রাখো।

# Multi-PDF RAG Assistant

A bilingual Retrieval-Augmented Generation (RAG) system for answering questions from multiple PDF documents using local and GPU-based LLM pipelines.

The project supports Bangla and English queries, automatic PDF indexing, source tracking, vector search, hybrid retrieval, and strict PDF-grounded answer generation.

---

## Main Concept

This project implements a PDF-based RAG pipeline.

The system extracts information from PDF files, converts the extracted content into embeddings, stores the embeddings in a vector database, retrieves relevant chunks based on user queries, and generates answers using only the retrieved PDF context.

If the answer is not available in the PDF content, the system returns `NOT FOUND`.

---

## How the RAG Pipeline Works

1. PDF files are loaded into the system.
2. Text is extracted from the PDF documents.
3. Extracted text is divided into smaller chunks.
4. Each chunk is converted into an embedding.
5. Embeddings are stored in ChromaDB.
6. When a user asks a question, the question is also converted into an embedding.
7. The system searches for similar chunks in the vector database.
8. Relevant chunks are passed to the LLM as context.
9. The LLM generates an answer using only the PDF information.
10. If no relevant answer is found, the system returns `NOT FOUND`.

---

## Architecture Overview

PDF Documents  
↓  
Text Extraction  
↓  
Chunking  
↓  
Embedding Generation  
↓  
Vector Database Storage  
↓  
User Query  
↓  
Query Embedding  
↓  
Relevant Chunk Retrieval  
↓  
Context Building  
↓  
LLM Answer Generation  

---

## Version 1: Local Pipeline

This version runs locally using VS Code and Ollama.

### Implementation Details

| Component | Implementation |
|---|---|
| PDF extraction | pdfplumber |
| Table extraction | Page text and tables extracted |
| Table handling | Tables extracted row by row |
| Chunking | RecursiveCharacterTextSplitter |
| Chunk size | 900 |
| Chunk overlap | 150 |
| Embedding model | paraphrase-multilingual-MiniLM-L12-v2 |
| Vector database | ChromaDB |
| Similarity metric | Cosine similarity |
| Retrieval method | 80% vector search + 20% BM25 |
| LLM | qwen2.5:7b-instruct-q4_K_M |
| Runtime | Local CPU via Ollama |
| Temperature | 0 |

### Metadata Stored Per Chunk

Each chunk stores:

- Source filename
- Page number
- Content type: text, table, or table row
- Table index
- Row index
- Chunk index

This metadata helps the system track where each answer comes from.

### Local Pipeline Features

- Extracts both page text and tables using pdfplumber
- Keeps table rows intact for precise retrieval
- Uses multilingual lightweight embeddings
- Runs locally on CPU
- Uses hybrid retrieval with vector search and BM25
- Applies threshold filtering
- Returns `NOT FOUND` if no chunk passes the threshold

---

## Version 2: Kaggle GPU Pipeline

This version runs on Kaggle GPU for stronger multilingual retrieval and faster inference.

### Implementation Details

| Component | Implementation |
|---|---|
| PDF extraction | pdfplumber |
| Table extraction | Page text only |
| Chunking | Custom character-based splitter |
| Chunk size | 1000 |
| Chunk overlap | 200 |
| Embedding model | BAAI/bge-m3 |
| Vector database | ChromaDB |
| LLM | Qwen2.5-7B |
| Quantization | 4-bit NF4 using BitsAndBytes |
| Runtime | Kaggle GPU |

### Retrieval Strategy

| Query Type | Retrieval Method |
|---|---|
| Bangla queries | Vector search only |
| English queries | Hybrid retrieval |
| English hybrid ratio | 85% vector + 15% BM25 |

BGE-M3 is used because it provides stronger multilingual embedding performance, especially for Bangla queries.

---

## Feature Details

| Feature | Description |
|---|---|
| Case-insensitive search | Uses casefold() normalization |
| PDF-only answers | Model is strictly prompted to use PDF context only |
| Auto detection | Detects new, updated, and removed PDF files |
| Auto indexing | Uses MD5 hash tracking |
| Bangla + English support | Supports bilingual queries |
| Source tracking | Tracks source file, page number, and content type |
| Strict fallback | Returns NOT FOUND when the answer is unavailable |

---

## Tech Stack

- Python
- pdfplumber
- ChromaDB
- BM25
- Sentence Transformers
- BAAI/bge-m3
- Ollama
- Qwen2.5-7B
- BitsAndBytes
- Kaggle GPU

---

## Use Cases

This project can be used for:

- Academic PDF question answering
- Research document search
- Bangla-English document retrieval
- Policy or guideline document QA
- Multi-PDF knowledge assistant systems
- Local private document chatbot systems

---

## Current Status

The project currently includes two working implementations:

- Local CPU-based pipeline using Ollama
- Kaggle GPU-based pipeline using BGE-M3 and Qwen2.5-7B

---

## Future Improvements

- Add OCR support for scanned PDFs
- Add a Streamlit or Gradio interface
- Improve table reconstruction
- Add citation-style answer display
- Add multi-document comparison
- Add retrieval evaluation metrics

---

## Author

Developed as a bilingual multi-PDF RAG project focused on Bangla and English document question answering.


PDF RAG Question Answering System
A Retrieval-Augmented Generation (RAG) system that answers natural language questions from PDF documents using hybrid search and a local LLM.

Tech Stack
Embedding Model: BGE-M3 (BAAI)
Vector Database: ChromaDB
LLM: Qwen 2.5-7B-Instruct (4-bit quantized)
Retrieval: Hybrid Search — BM25 + Dense Vector (weighted fusion)
UI: Gradio
Platform: Kaggle (GPU)
How It Works
PDF text is extracted and split into overlapping chunks
Each chunk is embedded using BGE-M3 and stored in ChromaDB
User query is embedded and matched via Hybrid Search (BM25 + vector)
Top chunks are passed as context to Qwen 2.5-7B
Model generates a grounded answer — returns NOT FOUND if answer is absent
Features
Multi-PDF upload and querying
Auto-detection & auto-indexing of new PDFs
Case-insensitive query handling
Bangla + English+ language and Table without Table support
Source tracking (file, page, relevance score)

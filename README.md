# PDF RAG Question Answering System

A Retrieval-Augmented Generation (RAG) system that answers natural language 
questions from PDF documents using hybrid search and a local LLM.

##  Tech Stack
- **Embedding Model:** BGE-M3 (BAAI)
- **Vector Database:** ChromaDB
- **LLM:** Qwen 2.5-7B-Instruct (4-bit quantized)
- **Retrieval:** Hybrid Search — BM25 + Dense Vector (weighted fusion)
- **UI:** Gradio
- **Platform:** Kaggle (GPU)

## How It Works
1. PDF text is extracted and split into overlapping chunks
2. Each chunk is embedded using BGE-M3 and stored in ChromaDB
3. User query is embedded and matched via Hybrid Search (BM25 + vector)
4. Top chunks are passed as context to Qwen 2.5-7B
5. Model generates a grounded answer — returns **NOT FOUND** if answer is absent

## Features 
- Multi-PDF upload and querying
- Auto-detection & auto-indexing of new PDFs
- Case-insensitive query handling
- Bangla + English+ language and Table without Table support
- Source tracking (file, page, relevance score)



                                                                  Main Concept
RAG implementation
1.META use
2.Extract text from PDF
3.Divide text into small chunks
4.Create vector by embedding each chunk and store in vector DB (Chroma)
5.When user asks a question, vectorize the question and search for similarity to find relevant chunks
6.If those chunks are given to LLM as context, LLM looks at the PDF information and gives answer (NOT FOUND if not)                                               

                                                        I implemented the RAG pipeline

Version 1  Local (VS Code + Ollama)
i. Metadata Usage Each chunk stores: source filename, page number, content type (text / table / table_row), table index, row index, chunk index.
ii. Text Extraction Extracted using  pdfplumber both page text and tables. Tables are extracted row-by-row for precise retrieval.
iii. Chunking  RecursiveCharacterTextSplitter  -chunk size 900, overlap 150. Separators include Bengali । character. Tables/rows kept intact.
iv. Embedding paraphrase-multilingual-MiniLM-L12-v2  lightweight, runs on CPU locally.
v. Vector Database ChromaDB persistent storage with cosine similarity space.
vi. Query Embedding Same model, normalized embedding.
vii. Retrieval Hybrid — 80% vector + 20% BM25. Threshold filtering applied.
viii. Context Building Source, page, content type included in context block.
ix. Answer Generation Ollama local server —qwen2.5:7b-instruct-q4_K_M.  Temperature 0 for deterministic answers.
x. NOT FOUND Returned if no chunk passes threshold or model finds no answer.
________________________________________
Version 2 (Kaggle + GPU)
i. Metadata Usage Source filename, page number, content type, section, chunk index.
ii. Text Extraction pdfplumber page text only.
iii. Chunking Custom character-based splitter — chunk size 1000, overlap 200.
iv. Embedding  BAAI/bge-m3 stronger multilingual model, runs on GPU.
v. Vector Database ChromaDB persistent storage.
vi. Query Embedding Same BGE-M3 model, normalized.
vii. Retrieval Bangla queries — vector only (no hard threshold, BGE-M3 handles it). English queries — hybrid, 85% vector + 15% BM25, threshold filtered.
viii. Context Building Same structured format with source and page.
ix. Answer Generation Qwen2.5-7B loaded with 4-bit quantization (NF4) via BitsAndBytes. Runs on Kaggle GPU.
x. NOT FOUND Same strict handling.


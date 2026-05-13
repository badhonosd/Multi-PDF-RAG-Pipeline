Main Concept

RAG implementation

META use
Extract text from PDF
Divide text into small chunks
Create vector by embedding each chunk and store in vector DB (Chroma)
When user asks a question, vectorize the question and search for similarity to find relevant chunks
If those chunks are given to LLM as context, LLM looks at the PDF information and gives answer (NOT FOUND if not)
iii. Chunking Custom character-based splitter — chunk size 1000, overlap 200.
iv. Embedding  BAAI/bge-m3 stronger multilingual model, runs on GPU.
v. Vector Database ChromaDB persistent storage.
vi. Query Embedding Same BGE-M3 model, normalized.
vii. Retrieval Bangla queries — vector only (no hard threshold, BGE-M3 handles it). English queries — hybrid, 85% vector + 15% BM25, threshold filtered.
viii. Context Building Same structured format with source and page.
ix. Answer Generation Qwen2.5-7B loaded with 4-bit quantization (NF4) via BitsAndBytes. Runs on Kaggle GPU.
x. NOT FOUND Same strict handling.


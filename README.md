# Main Concept

RAG implementation

1. META use
2. Extract text from PDF
3. Divide text into small chunks
4. Create vector by embedding each chunk and store in vector DB (Chroma)
5. When user asks a question, vectorize the question and search for similarity to find relevant chunks
6. If those chunks are given to LLM as context, LLM looks at the PDF information and gives answer (`NOT FOUND` if not)

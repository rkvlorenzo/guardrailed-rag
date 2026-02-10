# RAGuard
Dockerized **RAG (Retrieval-Augmented Generation)** system with persistent FAISS vector store, a CLI-based conversational interface, and evaluator-augmented LLM responses for reliable, grounded answers.

# Features

- Ingest **PDFs** and convert them into a persistent FAISS vector store
- **RAG pipeline** using LangChain + OpenAI embeddings for context-aware answers
- **Evaluator LLM** checks each response for:
  - `grounded` – is the answer supported by context?
  - `relevance` – does it answer the question?
  - `hallucination_risk` – likelihood of invented information
  - `confidence_score` – numeric confidence metric (0–1)
- **CLI interface** with Rich for colored output and loading spinners
- Docker-ready for quick deployment

## Getting Started

### 1. Clone the repo

```bash
https://github.com/rkvlorenzo/guardrailed-rag.git
cd guardrailed-rag
```

### 2. Create .env file
```bash
DOCS_FOLDER=docs
VECTOR_STORE_FOLDER=vector_store
OPENAI_API_KEY=<api_key>
```
### 3. Create docs and vector_store folders

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the CLI
```bash
python main.py
```
Type questions related to your documents. Use exit or quit to leave.
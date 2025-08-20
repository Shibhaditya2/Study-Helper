# Minimal Agentic RAG-Powered Study Helper

## Overview
A lightweight AI tutor built with Python and open-source models. Uses RAG with FAISS for document search (txt/pdf/epub), a SKOS/RDF knowledge graph for concept prerequisites, and FLAN-T5-Large for answering queries. Features a Streamlit UI with dynamic file upload.

- **Features**:
  - Ingests `.txt`, `.pdf`, `.epub` files.
  - Knowledge graph with concepts/prerequisites from `concepts.csv`.
  - FAISS vector search with `BAAI/bge-base-en-v1.5` embeddings.
  - Streamlit UI with chat and file upload.
  - Auto-builds indexes on startup.

## Setup
1. Create virtual env: `python -m venv study_helper_env` and activate it.
2. Install deps: `pip install -r requirements.txt`
3. Prepare data: Add `.txt`, `.pdf`, `.epub` files to `data/` and `concepts.csv` (columns: concept,prerequisite).
4. Run: `streamlit run app.py`

## Usage
- Open http://localhost:8501.
- Ask questions in the chat (e.g., "Explain Calculus").
- Upload new txt/pdf/epub files in sidebar to update vector store.

## Files
- `data/`: Input files (`.txt`, `.pdf`, `.epub`, `concepts.csv`).
- `ingest_documents.py`: Builds FAISS vector store.
- `build_knowledge_graph.py`: Builds RDF KG.
- `kg_retriever.py`: KG traversal.
- `app.py`: Streamlit UI with query processing.
- `requirements.txt`: Dependencies.
- `README.md`: This file.

## Notes
- Runs on CPU; GPU optional.
- Expand with more documents or concepts in `concepts.csv`.
- Built with open-source: HuggingFace, LangChain, Streamlit.
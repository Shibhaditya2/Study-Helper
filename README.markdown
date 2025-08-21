Agentic RAG-Powered Study Helper (Final Version with GPU)
Overview
Local AI tutor with RAG, KG, agents, and enhanced Streamlit UI. Uses GPU by default for LLM and embeddings, with CPU fallback. Integrates docs/web, solves math/plots, generates quizzes/flashcards, logs stats.

Timeline: [May 2025 - Jun 2025]
Features:
Ingests .txt, .pdf, .epub files.
Auto-generated 1,200+ concepts KG.
FAISS vector search with BAAI/bge-base-en-v1.5 (GPU-accelerated).
Agents for math (SymPy), plotting (Matplotlib), web search (DuckDuckGo).
Quiz/flashcard generation.
Streamlit UI with tabs, dynamic upload, stats display (including GPU).
Logging: Vector store/KG build stats, query/quiz/flashcard times, upload details, GPU usage.



Setup

Install NVIDIA Drivers and CUDA:
Ensure an NVIDIA GPU with CUDA support (https://developer.nvidia.com/cuda-downloads).
Install CUDA Toolkit 12.1 or compatible.


Create virtual env: python -m venv study_helper_env and activate.
Install: pip install -r requirements.txt
Add docs to data/ (optional; samples included).
Run: streamlit run streamlit_app.py

Usage

Chat tab: Ask questions (e.g., "Explain Calculus", "Solve x^2 - 4 = 0").
Quiz/Flashcards tabs: Generate content by topic/concept.
Upload/Settings tab: Upload files, rebuild KG, view stats (including GPU).
Logs: Check logs/app.log and console for stats (build times, document counts, GPU info).

Files

data/: Docs and concepts.csv (auto-generated).
ingest_documents.py: Builds FAISS vector store (GPU), logs stats.
build_knowledge_graph.py: Builds RDF KG, logs stats.
generate_concepts.py: Generates concepts.csv (1,200+ concepts).
kg_retriever.py: KG traversal.
agents.py: Agents, tools, quiz/flashcard generation, query logging, GPU support.
streamlit_app.py: Streamlit UI with tabs, stats display.
requirements.txt: Dependencies (includes CUDA).
.gitignore: Excludes generated files, logs.
README.md: This file.

Notes

GPU: Uses CUDA by default; falls back to CPU if no GPU. Requires NVIDIA GPU, CUDA 12.1, cuDNN.
Python: Use Python 3.11 for compatibility.
Logs: In logs/app.log and console (includes GPU stats).
Expand concepts in generate_concepts.py.
Built with open-source: HuggingFace, LangChain, Streamlit.

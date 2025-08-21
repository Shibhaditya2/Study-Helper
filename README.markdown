# Agentic RAG-Powered Study Helper (Final Version with GPU)

## 📖 Overview
A **local AI tutor** powered by Retrieval-Augmented Generation (RAG), Knowledge Graphs (KG), and agents, with an enhanced **Streamlit UI**.  
- Uses **GPU by default** for LLMs and embeddings (with CPU fallback).  
- Integrates documents and web search.  
- Solves math problems, generates plots, quizzes, and flashcards.  
- Logs stats for transparency and debugging.  

**Timeline:** *May 2025 – Jun 2025*

---

## ✨ Features
- Ingests `.txt`, `.pdf`, `.epub` files.  
- Auto-generated **1,200+ concepts KG**.  
- **FAISS vector search** with `BAAI/bge-base-en-v1.5` (GPU-accelerated).  
- Agents for:
  - **Math** → SymPy  
  - **Plotting** → Matplotlib  
  - **Web Search** → DuckDuckGo  
- **Quiz & Flashcard** generation.  
- **Streamlit UI**:
  - Tabs for chat, quizzes, flashcards, uploads, and stats.  
  - Dynamic upload + rebuild options.  
  - Displays **GPU usage**.  
- **Logging**:
  - Vector store/KG build stats.  
  - Query/quiz/flashcard timings.  
  - Upload details.  
  - GPU stats.  

---

## ⚙️ Setup

### 1. Install NVIDIA Drivers & CUDA
- Ensure an **NVIDIA GPU** with CUDA support → [Download CUDA](https://developer.nvidia.com/cuda-downloads).  
- Install **CUDA Toolkit 12.1** (or compatible).  

### 2. Create Virtual Environment
```bash
python -m venv study_helper_env
study_helper_env\Scripts\activate  # Windows
source study_helper_env/bin/activate  # Linux/Mac

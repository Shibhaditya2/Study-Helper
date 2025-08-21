import os
import time
import logging
import torch
import streamlit as st
from ingest_documents import ingest_docs
from build_knowledge_graph import build_kg
from generate_concepts import generate_concepts
from agents import process_query, generate_quiz, generate_flashcards, load_vector_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Check GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    logger.info(f"Application starting with GPU: {gpu_name}, total memory: {gpu_memory:.2f}GB")
else:
    logger.info("Application starting with CPU (no GPU detected)")

# Auto-generate concepts.csv
if not os.path.exists("data/concepts.csv"):
    logger.info("Generating concepts.csv")
    generate_concepts()

# Auto-build indexes
if not os.path.exists("faiss_index"):
    logger.info("Starting vector store build")
    ingest_docs()
if not os.path.exists("knowledge_graph.ttl"):
    logger.info("Starting knowledge graph build")
    build_kg()

# Reload vector store
from agents import vector_store
vector_store = load_vector_store()
if not vector_store:
    st.error("Failed to load vector store. Please rebuild it in the Upload/Settings tab.")

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f0f2f6;}
    .stTabs [data-testid="stTab"] {background-color: #ffffff; border-radius: 5px; padding: 8px 16px;}
    .stTabs [aria-selected="true"] {background-color: #4caf50; color: white;}
    .chat-message {border-radius: 10px; padding: 10px; margin-bottom: 10px;}
    .user-message {background-color: #dcf8c6;}
    .assistant-message {background-color: #ffffff;}
    .stats-box {background-color: #e3f2fd; padding: 10px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

st.title("Agentic RAG-Powered Study Helper")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = {"docs_ingested": 0, "chunks": 0, "index_size": 0, "concepts": 0, "triples": 0, "kg_size": 0, "device": device, "gpu_name": gpu_name if device == "cuda" else "CPU", "gpu_memory": gpu_memory if device == "cuda" else 0}

tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Quiz", "Flashcards", "Upload/Settings"])

with tab1:
    for message in st.session_state.messages:
        with st.container():
            if message["role"] == "user":
                st.markdown(f"<div class='chat-message user-message'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-message assistant-message'>{message['content']}</div>", unsafe_allow_html=True)

    if prompt := st.chat_input("Ask a question..."):
        logger.info(f"User query: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            state = {"query": prompt}
            response = process_query(state)
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
        st.rerun()

with tab2:
    topic = st.text_input("Enter topic for quiz:")
    if st.button("Generate Quiz"):
        with st.spinner("Generating..."):
            quiz = generate_quiz(topic)
        st.markdown(quiz)

with tab3:
    concept = st.text_input("Enter concept for flashcards:")
    if st.button("Generate Flashcards"):
        with st.spinner("Generating..."):
            flashcards = generate_flashcards(concept)
        st.markdown(flashcards)

with tab4:
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader("Upload txt/pdf/epub", type=["txt", "pdf", "epub"], accept_multiple_files=True)
    if uploaded_files:
        start_time = time.time()
        file_count = len(uploaded_files)
        total_size = sum(file.size for file in uploaded_files) / 1024 / 1024  # MB
        logger.info(f"Uploading {file_count} files, total size={total_size:.2f}MB")
        for file in uploaded_files:
            file_path = os.path.join("data", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        with st.spinner("Rebuilding vector store..."):
            ingest_docs(force_rebuild=True)
            vector_store = load_vector_store()
            if vector_store:
                st.session_state.stats["docs_ingested"] += file_count
                st.session_state.stats["index_size"] = sum(os.path.getsize(os.path.join("faiss_index", f)) for f in os.listdir("faiss_index") if os.path.isfile(os.path.join("faiss_index", f))) / 1024 / 1024
                st.success(f"Vector store updated! {file_count} files uploaded ({total_size:.2f}MB)")
            else:
                st.error("Failed to rebuild vector store. Check logs for details.")
        elapsed_time = time.time() - start_time
        logger.info(f"Upload and rebuild completed: time={elapsed_time:.2f}s")

    st.subheader("Rebuild Knowledge Graph")
    if st.button("Rebuild Knowledge Graph"):
        with st.spinner("Rebuilding KG..."):
            generate_concepts()
            build_kg()
            st.session_state.stats["concepts"] = sum(1 for _ in open("data/concepts.csv")) - 1  # Exclude header
            st.session_state.stats["kg_size"] = os.path.getsize("knowledge_graph.ttl") / 1024 / 1024
        st.success("Knowledge graph rebuilt!")

    st.subheader("System Stats")
    if not vector_store:
        st.warning("Vector store not loaded. Upload documents or rebuild to enable full functionality.")
    st.markdown(f"""
    <div class='stats-box'>
        <b>Device:</b> {st.session_state.stats['device'].upper()} ({st.session_state.stats['gpu_name']})<br>
        <b>GPU Memory:</b> {st.session_state.stats['gpu_memory']:.2f}GB<br>
        <b>Vector Store:</b> {st.session_state.stats['docs_ingested']} documents, {st.session_state.stats['chunks']} chunks, {st.session_state.stats['index_size']:.2f}MB<br>
        <b>Knowledge Graph:</b> {st.session_state.stats['concepts']} concepts, {st.session_state.stats['triples']} triples, {st.session_state.stats['kg_size']:.2f}MB
    </div>
    """, unsafe_allow_html=True)
import os
import time	
import logging
import torch
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

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
    logger.info(f"Using GPU: {gpu_name}, total memory: {gpu_memory:.2f}GB")
else:
    logger.info("No GPU detected; falling back to CPU")

# Initialize embeddings with GPU
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={"device": device})

def extract_epub(file_path):
    logger.info(f"Extracting EPUB: {file_path}")
    try:
        book = epub.read_epub(file_path)
        text = ""
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            text += soup.get_text(separator=" ", strip=True) + " "
        return [Document(page_content=text, metadata={"source": file_path})]
    except Exception as e:
        logger.error(f"Failed to extract EPUB {file_path}: {e}")
        return []

def ingest_docs(data_folder="data", force_rebuild=False):
    start_time = time.time()
    logger.info(f"Starting document ingestion (force_rebuild={force_rebuild}, device={device})")
    
    if force_rebuild and os.path.exists("faiss_index"):
        logger.info("Removing existing FAISS index")
        import shutil
        shutil.rmtree("faiss_index")
    
    if os.path.exists("faiss_index") and not force_rebuild:
        logger.info("FAISS index exists; skipping rebuild")
        return
    
    documents = []
    file_count = {"txt": 0, "pdf": 0, "epub": 0}
    for file in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file)
        try:
            if file.endswith(".txt"):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
                file_count["txt"] += 1
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
                file_count["pdf"] += 1
            elif file.endswith(".epub"):
                documents.extend(extract_epub(file_path))
                file_count["epub"] += 1
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
    
    if not documents:
        logger.warning("No documents found in data folder")
        return
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(documents)
    
    vector_store = FAISS.from_documents(splits, embeddings)
    vector_store.save_local("faiss_index")
    
    # Log stats
    elapsed_time = time.time() - start_time
    index_size = sum(os.path.getsize(os.path.join("faiss_index", f)) for f in os.listdir("faiss_index") if os.path.isfile(os.path.join("faiss_index", f))) / 1024 / 1024  # MB
    logger.info(f"Vector store built: {len(documents)} documents, {len(splits)} chunks, {file_count['txt']} txt, {file_count['pdf']} pdf, {file_count['epub']} epub, time={elapsed_time:.2f}s, index_size={index_size:.2f}MB, device={device}")
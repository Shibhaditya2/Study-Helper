import os
import warnings
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
try:
	from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
	from langchain_community.embeddings import HuggingFaceEmbeddings

# Suppress noisy warnings (numpy on Windows 3.13)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

def extract_epub(file_path):
	book = epub.read_epub(file_path)
	text = ""
	for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
		soup = BeautifulSoup(item.get_content(), "html.parser")
		text += soup.get_text(separator=" ", strip=True) + " "
	return [Document(page_content=text, metadata={"source": file_path})]

def ingest_docs(data_folder="data", force_rebuild=False):
	if force_rebuild and os.path.exists("faiss_index"):
		import shutil
		shutil.rmtree("faiss_index")
	
	if os.path.exists("faiss_index") and not force_rebuild:
		print("FAISS index exists.")
		return
	
	documents = []
	for file in os.listdir(data_folder):
		file_path = os.path.join(data_folder, file)
		try:
			if file.endswith(".txt"):
				loader = TextLoader(file_path)
				documents.extend(loader.load())
			elif file.endswith(".pdf"):
				loader = PyPDFLoader(file_path)
				documents.extend(loader.load())
			elif file.endswith(".epub"):
				documents.extend(extract_epub(file_path))
		except Exception as e:
			print(f"Error processing {file}: {e}")
	
	if not documents:
		print("No documents found.")
		return
	
	splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
	splits = splitter.split_documents(documents)
	
	# Create a SMALL embedding model to reduce memory footprint
	try:
		embeddings = HuggingFaceEmbeddings(
			model_name="sentence-transformers/all-MiniLM-L6-v2",
			model_kwargs={"device": "cpu"}
		)
		vector_store = FAISS.from_documents(splits, embeddings)
		vector_store.save_local("faiss_index")
		print("Documents ingested into FAISS.")
	except Exception as e:
		print(f"Error building vector store: {e}")
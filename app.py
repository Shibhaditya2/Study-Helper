import os
import warnings
import streamlit as st

# Suppress numpy warnings that cause crashes on Python 3.13 Windows
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

MODULES_LOADED = False
llm = None
embeddings = None

try:
	from langchain_huggingface import HuggingFacePipeline
	from langchain_core.prompts import PromptTemplate
	from langchain_community.vectorstores import FAISS
	try:
		from langchain_huggingface import HuggingFaceEmbeddings as HFEmb
	except Exception:
		from langchain_community.embeddings import HuggingFaceEmbeddings as HFEmb
	from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
	from ingest_documents import ingest_docs
	from build_knowledge_graph import build_kg
	from kg_retriever import load_kg, traverse_kg

	# Initialize a SMALLER LLM to reduce memory footprint
	# Use t5-small as a fallback if flan-t5-large is too big
	try:
		tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
		model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
	except Exception:
		tokenizer = AutoTokenizer.from_pretrained("t5-small")
		model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

	flan_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=256)
	llm = HuggingFacePipeline(pipeline=flan_pipeline)

	# Initialize SMALL embeddings to reduce memory pressure
	embeddings = HFEmb(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

	MODULES_LOADED = True
except Exception as e:
	st.error(f"Error loading AI modules: {e}")
	MODULES_LOADED = False
	llm = None
	embeddings = None


def load_vector_store():
	if not MODULES_LOADED or not embeddings:
		return None
	try:
		return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
	except Exception as e:
		print(f"Error loading vector store: {e}")
		return None

# Auto-build indexes only if modules are loaded
if MODULES_LOADED:
	if not os.path.exists("faiss_index"):
		st.info("Building vector store...")
		try:
			ingest_docs()
		except Exception as e:
			st.error(f"Error building vector store: {e}")
	if not os.path.exists("knowledge_graph.ttl"):
		st.info("Building knowledge graph...")
		try:
			build_kg()
		except Exception as e:
			st.error(f"Error building knowledge graph: {e}")

vector_store = load_vector_store()
kg = load_kg() if MODULES_LOADED else None

# Prompt
if MODULES_LOADED:
	rag_prompt = PromptTemplate(
		input_variables=["question", "context"],
		template="Question: {question}\nContext: {context}\nAnswer:"
	)
else:
	rag_prompt = None

# Streamlit UI
st.title("Study Helper")

# Status indicator
st.header("System Status")
col1, col2 = st.columns(2)

with col1:
	if MODULES_LOADED:
		st.success("✅ AI Modules: Loaded")
		if llm:
			st.success("✅ Language Model: Ready (small)")
		else:
			st.error("❌ Language Model: Failed")
		if embeddings:
			st.success("✅ Embeddings: Ready (MiniLM)")
		else:
			st.error("❌ Embeddings: Failed")
	else:
		st.error("❌ AI Modules: Failed to load")

with col2:
	if vector_store:
		st.success("✅ Vector Store: Available")
	else:
		st.warning("⚠️ Vector Store: Not built")
	if kg:
		st.success("✅ Knowledge Graph: Available")
	else:
		st.warning("⚠️ Knowledge Graph: Not built")

st.divider()

# Sidebar for file upload
st.sidebar.title("Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload txt/pdf/epub", type=["txt", "pdf", "epub"], accept_multiple_files=True)
if uploaded_files and MODULES_LOADED:
	for file in uploaded_files:
		file_path = os.path.join("data", file.name)
		with open(file_path, "wb") as f:
			f.write(file.getbuffer())
	st.sidebar.success("Files uploaded! Rebuilding vector store...")
	try:
		ingest_docs(force_rebuild=True)
		vector_store = load_vector_store()
		st.sidebar.success("Vector store updated.")
	except Exception as e:
		st.sidebar.error(f"Error updating vector store: {e}")
elif uploaded_files and not MODULES_LOADED:
	st.sidebar.warning("AI modules not loaded. Cannot process uploaded files.")

# Chat interface
if "messages" not in st.session_state:
	st.session_state.messages = []

for message in st.session_state.messages:
	with st.chat_message(message["role"]):
		st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
	st.session_state.messages.append({"role": "user", "content": prompt})
	with st.chat_message("user"):
		st.markdown(prompt)
	
	with st.chat_message("assistant"):
		if not MODULES_LOADED:
			response = "AI processing is not available. Please check the error messages above."
		elif not vector_store or not kg:
			response = "Vector store or knowledge graph not available. Please wait for them to be built."
		elif not llm:
			response = "Language model not available. Please check the error messages above."
		else:
			with st.spinner("Thinking..."):
				try:
					docs = vector_store.similarity_search(prompt, k=3)
					docs_content = "\n".join([doc.page_content for doc in docs])
					prereqs = traverse_kg(kg, prompt.split()[-1])
					context = f"{docs_content}\nPrerequisites: {prereqs}"
					input_text = rag_prompt.format(question=prompt, context=context)
					response = llm(input_text)
				except Exception as e:
					response = f"Error processing your question: {e}"
		st.markdown(response)
	st.session_state.messages.append({"role": "assistant", "content": response})
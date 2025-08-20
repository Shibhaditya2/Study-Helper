import os
import streamlit as st
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from ingest_documents import ingest_docs
from build_knowledge_graph import build_kg
from kg_retriever import load_kg, traverse_kg

# Initialize LLM
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
flan_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=flan_pipeline)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={"device": "cpu"})

def load_vector_store():
    try:
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

# Auto-build indexes
if not os.path.exists("faiss_index"):
    st.info("Building vector store...")
    ingest_docs()
if not os.path.exists("knowledge_graph.ttl"):
    st.info("Building knowledge graph...")
    build_kg()

vector_store = load_vector_store()
kg = load_kg()

# Prompt
rag_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="Question: {question}\nContext: {context}\nAnswer:"
)

# Streamlit UI
st.title("Study Helper")

# Sidebar for file upload
st.sidebar.title("Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload txt/pdf/epub", type=["txt", "pdf", "epub"], accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join("data", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success("Files uploaded! Rebuilding vector store...")
    ingest_docs(force_rebuild=True)
    global vector_store
    vector_store = load_vector_store()
    st.sidebar.success("Vector store updated.")

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
        with st.spinner("Thinking..."):
            if vector_store and kg:
                docs = vector_store.similarity_search(prompt, k=3)
                docs_content = "\n".join([doc.page_content for doc in docs])
                prereqs = traverse_kg(kg, prompt.split()[-1])
                context = f"{docs_content}\nPrerequisites: {prereqs}"
                input_text = rag_prompt.format(question=prompt, context=context)
                response = llm(input_text)
            else:
                response = "Error: Vector store or knowledge graph not initialized."
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
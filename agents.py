import time
import logging
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt
import sympy
from kg_retriever import load_kg, traverse_kg

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
    logger.info(f"Using GPU for LLM and embeddings: {gpu_name}, total memory: {gpu_memory:.2f}GB")
else:
    logger.info("No GPU detected; falling back to CPU for LLM and embeddings")

# Initialize LLM with GPU
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large").to(device)
flan_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512, device=0 if device == "cuda" else -1)
llm = HuggingFacePipeline(pipeline=flan_pipeline)

# Initialize embeddings with GPU
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={"device": device})

def load_vector_store():
    try:
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        # Log embedding dimensionality
        sample_text = "test"
        sample_embedding = embeddings.embed_query(sample_text)
        logger.info(f"Embedding dimensionality: {len(sample_embedding)}")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store: {e}")
        return None

vector_store = load_vector_store()
kg = load_kg()

python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="Execute Python code for math (SymPy) or plots (Matplotlib). Input: valid Python code. Print outputs.",
    func=python_repl.run,
)

search_tool = DuckDuckGoSearchRun(name="web_search", description="Search web for current info.")

def kg_tool(input):
    return str(traverse_kg(kg, input))

kg_tool = Tool(
    name="kg_traversal",
    description="Traverse knowledge graph for prerequisites. Input: concept name.",
    func=kg_tool,
)

retriever_tool = Tool(
    name="document_retriever",
    description="Retrieve from documents. Input: query.",
    func=lambda q: "\n".join([doc.page_content for doc in vector_store.similarity_search(q, k=3)]) if vector_store else "Vector store not loaded",
)

tools = [repl_tool, search_tool, kg_tool, retriever_tool]

rag_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="Question: {question}\nContext: {context}\nAnswer:"
)

def classify_query(query):
    if "solve" in query.lower() or "equation" in query.lower():
        return "math"
    elif "plot" in query.lower() or "graph" in query.lower():
        return "plot"
    elif "explain" in query.lower() or "concept" in query.lower():
        return "explanation"
    else:
        return "general"

def select_tool(query, category):
    """Select appropriate tool based on query category."""
    if category == "math":
        return [repl_tool]
    elif category == "plot":
        return [repl_tool]
    elif category == "explanation":
        return [retriever_tool, kg_tool]
    else:
        return [search_tool, retriever_tool]

def process_query(state):
    start_time = time.time()
    query = state["query"]
    logger.info(f"Processing query: {query}, device={device}")
    category = classify_query(query)
    
    if category in ["math", "plot"]:
        # Generate Python code for math or plot queries
        if category == "math":
            code = f"import sympy; print(sympy.solve({query}))"
        else:  # plot
            code = f"import matplotlib.pyplot as plt; x = range(10); y = [i**2 for i in x]; plt.plot(x, y); plt.show()"
        result = repl_tool.func(code)
    else:
        # Use retriever and KG for explanation/general queries
        tools_to_use = select_tool(query, category)
        context_parts = []
        for tool in tools_to_use:
            try:
                if tool.name == "kg_traversal":
                    context_parts.append(f"Prerequisites: {tool.func(query.split()[-1])}")
                else:
                    result = tool.func(query)
                    if result == "Vector store not loaded":
                        logger.error("Vector store not loaded; cannot retrieve documents")
                        context_parts.append("No document context available")
                    else:
                        context_parts.append(result)
            except Exception as e:
                logger.error(f"Error using tool {tool.name}: {e}")
                context_parts.append(f"Error with {tool.name}: {e}")
        context = "\n".join(context_parts)
        input_prompt = rag_prompt.format(question=query, context=context)
        result = llm(input_prompt)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Query processed: category={category}, time={elapsed_time:.2f}s, device={device}")
    return {"answer": result}

quiz_prompt = PromptTemplate(
    input_variables=["topic", "context"],
    template="Generate 5 multiple-choice questions on {topic} with 4 options each and correct answers. Context: {context}\nFormat: Q1: question\nA) opt1 B) opt2 C) opt3 D) opt4\nAnswer: correct\n"
)

def generate_quiz(topic):
    start_time = time.time()
    logger.info(f"Generating quiz for topic: {topic}, device={device}")
    if not vector_store:
        logger.error("Vector store not loaded; cannot generate quiz")
        return "Error: Vector store not loaded"
    docs = retriever_tool.func(topic)
    if docs == "Vector store not loaded":
        logger.error("Vector store not loaded; cannot retrieve documents for quiz")
        return "Error: No document context available"
    input = quiz_prompt.format(topic=topic, context=docs)
    result = llm(input)
    elapsed_time = time.time() - start_time
    logger.info(f"Quiz generated: time={elapsed_time:.2f}s, device={device}")
    return result

flashcard_prompt = PromptTemplate(
    input_variables=["concept", "context"],
    template="Generate 5 flashcards for {concept}. Format: Front: question/concept\nBack: explanation\nContext: {context}\n"
)

def generate_flashcards(concept):
    start_time = time.time()
    logger.info(f"Generating flashcards for concept: {concept}, device={device}")
    if not vector_store:
        logger.error("Vector store not loaded; cannot generate flashcards")
        return "Error: Vector store not loaded"
    docs = retriever_tool.func(concept)
    if docs == "Vector store not loaded":
        logger.error("Vector store not loaded; cannot retrieve documents for flashcards")
        return "Error: No document context available"
    input = flashcard_prompt.format(concept=concept, context=docs)
    result = llm(input)
    elapsed_time = time.time() - start_time
    logger.info(f"Flashcards generated: time={elapsed_time:.2f}s, device={device}")
    return result
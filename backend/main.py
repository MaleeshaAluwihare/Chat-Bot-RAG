from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFPlumberLoader
from langchain_mongodb import MongoDBAtlasVectorSearch
from dotenv import load_dotenv
from pymongo import MongoClient
import os

load_dotenv()

DB_NAME = "Chatbot"
COLLECTION_NAME = "ChatBot"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "langchain-test-index-vectorstores"
EMBEDDING_KEY = "embedding"

# Initialize Embeddings
embeddings = OllamaEmbeddings(model="deepseek-r1:14b")

# Connect to MongoDB
MONGODB_URI = os.getenv("MONGODB_URI")
if not MONGODB_URI:
    raise ValueError("MongoDB URI not found in environment variables")

try:
    client = MongoClient(MONGODB_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print("Connected to MongoDB successfully")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit(1)

# Initialize MongoDB Vector Store
vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collection,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    text_key="text",  # Ensure the document stores raw text under this key
    embedding_key=EMBEDDING_KEY,
    relevance_score_fn="cosine",
)

# Initialize LLM
model = OllamaLLM(model="deepseek-r1:14b")

# Prompt Template
template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

Question: {question}
Context: {context}
Answer:
"""

pdf_directory = "./pdfs/"

def upload_pdf(file):
    """Uploads the PDF file to the server."""
    with open(os.path.join(pdf_directory, file.name), "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path):
    """Loads text from a PDF file."""
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents


def split_text(documents):
    """Splits text into chunks for vector storage."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)


def index_docs(documents):
    """Indexes the document chunks in the vector store."""
    formatted_docs = [
        {"text": doc.page_content, EMBEDDING_KEY: embeddings.embed_query(doc.page_content)}
        for doc in documents
    ]
    vector_store.add_documents(formatted_docs)


def retrieve_docs(query):
    """Retrieves relevant documents using similarity search."""
    return vector_store.similarity_search(query)


def normal_chat(question):
    """Handles general chatbot conversations."""
    return model.invoke(question)


def answer(question):
    """Determines whether to use RAG or normal chat based on document retrieval."""
    documents = retrieve_docs(question)
    if documents:
        context = "\n\n".join([doc["text"] for doc in documents])
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model
        return chain.invoke({"question": question, "context": context})
    else:
        return normal_chat(question)  # Default to normal chat if no context is found

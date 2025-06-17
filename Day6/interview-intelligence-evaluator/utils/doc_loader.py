import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DATA_DIR = "data/interview_docs"

def load_documents():
    loaders = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".txt"):
            loaders.append(TextLoader(os.path.join(DATA_DIR, fname)))
    return loaders

def get_vector_store():
    docs = []
    for loader in load_documents():
        docs.extend(loader.load())

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embedding = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    vector_store = FAISS.from_documents(split_docs, embedding)
    return vector_store

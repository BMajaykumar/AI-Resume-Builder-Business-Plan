from langchain.vectorstores import FAISS
from .preprocess import embeddings

VECTOR_DB_PATH = "vectorstore/index"

def load_vectorstore():
    return FAISS.load_local(VECTOR_DB_PATH, embeddings)

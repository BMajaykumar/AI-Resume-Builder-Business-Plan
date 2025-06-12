import os
from langchain_community.vectorstores import FAISS
from .preprocess import embeddings, get_chunked_docs

VECTOR_DB_PATH = "vectorstore/index"

def load_vectorstore():
    faiss_index_path = os.path.join(VECTOR_DB_PATH, "index.faiss")
    if os.path.exists(faiss_index_path):
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("[INFO] Vectorstore missing. Creating new FAISS index...")
        docs = get_chunked_docs()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(VECTOR_DB_PATH)
        return vectorstore

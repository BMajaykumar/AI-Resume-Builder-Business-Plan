from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

DATA_FOLDER = "data/"
VECTOR_DB_PATH = "vectorstore/index"

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def preprocess_documents():
    all_docs = []
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_FOLDER, file))
            docs = loader.load()
            all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_docs)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    print("Documents loaded and vectorstore saved.")

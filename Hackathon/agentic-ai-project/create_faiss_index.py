import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import VECTORSTORE_PATH, EMBEDDING_MODEL

DATA_DIR = "data/"
def create_faiss_index():
         embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
         documents = []
         for filename in os.listdir(DATA_DIR):
             if filename.endswith(".pdf"):
                 filepath = os.path.join(DATA_DIR, filename)
                 loader = PyPDFLoader(filepath)
                 documents.extend(loader.load())

         if not documents:
             raise ValueError("No PDF documents found in data/ directory. Please add PDFs to process.")

         text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
         docs = text_splitter.split_documents(documents)

         try:
             vectorstore = FAISS.from_documents(docs, embeddings)
             vectorstore.save_local(VECTORSTORE_PATH)
             print(f"FAISS vector store created and saved at {VECTORSTORE_PATH}")
         except Exception as e:
             raise Exception(f"Failed to create FAISS vector store: {e}")

if __name__ == "__main__":
         create_faiss_index()
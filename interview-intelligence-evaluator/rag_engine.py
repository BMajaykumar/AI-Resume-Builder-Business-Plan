from dotenv import load_dotenv
import os
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.doc_loader import get_vector_store

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def evaluate_with_rag(transcript):
    retriever = get_vector_store().as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = qa.run(transcript)
    
    return {"rag_response": response}

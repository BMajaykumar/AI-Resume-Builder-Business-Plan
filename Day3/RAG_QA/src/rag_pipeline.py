from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from .retriever import load_vectorstore

import os
from dotenv import load_dotenv
load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2
)

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

def answer_question(question):
    result = qa_chain({"query": question})
    return result['result'], result['source_documents']

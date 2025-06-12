from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from .retriever import load_vectorstore

llm = ChatOpenAI(temperature=0, model_name="gemini-2.0-flash")

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

def answer_question(question):
    result = qa_chain({"query": question})
    return result['result'], result['source_documents']

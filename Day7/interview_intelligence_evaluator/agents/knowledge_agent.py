import os
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import get_gemini_model

def create_vector_store():
    if not os.path.exists("./data/knowledge_base/interview_prep.txt"):
        raise FileNotFoundError("Missing knowledge base: interview_prep.txt")

    loader = TextLoader("./data/knowledge_base/interview_prep.txt")
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.path.exists("chroma_store"):
        return Chroma(persist_directory="chroma_store", embedding_function=embeddings)

    db = Chroma.from_documents(docs, embeddings, persist_directory="chroma_store")
    db.persist()
    return db

def validate_knowledge(transcript, vectorstore):
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(transcript)
    context = "\n".join([doc.page_content for doc in docs])
    model = get_gemini_model()
    prompt = f"""You are a domain expert. Evaluate the candidate's transcript against this context:

Context:
{context}

Transcript:
{transcript}
"""
    return model.generate_content(prompt).text

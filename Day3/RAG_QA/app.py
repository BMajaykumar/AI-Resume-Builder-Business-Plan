import streamlit as st
from src.rag_pipeline import answer_question

st.title("RAG QA System - AI Research Papers (Gemini)")
query = st.text_input("Ask a question about AI research papers")

if query:
    answer, sources = answer_question(query)
    st.write("### Answer:", answer)
    st.write("### Sources:")
    for doc in sources:
        st.write(f"**Page**: {doc.metadata.get('page', 'N/A')} - **Content**: {doc.page_content[:300]}...")

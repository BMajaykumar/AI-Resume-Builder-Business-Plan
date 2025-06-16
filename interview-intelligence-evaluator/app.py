# app.py
import streamlit as st
from utils.doc_loader import load_documents
from analyzer import analyze_transcript
from rag_engine import evaluate_with_rag
from generator import generate_report
import json

st.set_page_config(page_title="Interview Intelligence Evaluator")
st.title("Interview Intelligence Evaluator")

uploaded_file = st.file_uploader("Upload Transcript (.txt)", type=["txt"])

if uploaded_file is not None:
    transcript_text = uploaded_file.read().decode("utf-8")
    st.subheader("Transcript Preview")
    st.text_area("Transcript Content", transcript_text, height=200)

    with st.spinner("Analyzing interview..."):
        analysis = analyze_transcript(transcript_text)
        rag_results = evaluate_with_rag(transcript_text)
        final_report = generate_report(analysis, rag_results)

    st.success("Analysis Complete!")
    st.subheader("Final Evaluation Report")
    st.json(final_report)

    with open("reports/report_output.json", "w") as f:
        json.dump(final_report, f, indent=2)
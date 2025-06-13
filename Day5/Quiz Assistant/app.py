import streamlit as st
from utils.pdf_loader import load_pdf
from utils.summarizer import summarize_text
from utils.question_generator import generate_mcqs

def run():
    # --- Page Config ---
    st.set_page_config(
        page_title=" Study Assistant | Quiz Generator",
        layout="centered"
    )

    # --- Header Section ---
    st.markdown("""
        <style>
        .centered-title {
            text-align: center;
            font-size: 40px;
            font-weight: 700;
            color: #4A6FA5;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #555;
        }
        .stTextArea > label {
            font-weight: 600;
        }
        </style>
        <div class="centered-title">Study Assistant</div>
        <div class="subtitle">Summarize any PDF and auto-generate quiz questions from the content.</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- File Upload ---
    uploaded_file = st.file_uploader(" Upload your study material (PDF)", type=["pdf"])

    if uploaded_file is not None:
        with st.spinner(" Extracting text from PDF..."):
            text = load_pdf(uploaded_file)

        # --- Summary Section ---
        st.markdown("##  Summary of the PDF")
        with st.spinner(" Summarizing content..."):
            summary = summarize_text(text)

        st.text_area(" Summary", summary, height=200, key="summary_area")

        st.markdown("---")

        st.subheader(" Generate Quiz Questions")
        num_questions = st.number_input("How many MCQs do you want to generate?", min_value=5, max_value=10, step=5)

        if st.button("Generate MCQs"):
            with st.spinner("🎓 Creating quiz questions..."):
                questions = generate_mcqs(summary, num_questions)

            st.text_area("Quiz Questions (MCQs)", questions, height=300, key="quiz_area")
            st.success("Quiz ready! Use this to test your understanding. 💡")
    else:
        st.info("Please upload a PDF file to get started.")

if __name__ == "__main__":
    run()

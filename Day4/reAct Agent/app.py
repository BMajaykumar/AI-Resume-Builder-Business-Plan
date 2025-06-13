import streamlit as st
from agent import ReActAgent
from report_generator import generate_report

# --- Page Configuration ---
st.set_page_config(page_title="🧠 Web Research Agent", layout="centered")

# --- Header Section ---
st.markdown(
    """
    <h1 style='text-align: center; color: #4A6FA5;'>🔍 Web Research Agent</h1>
    <p style='text-align: center; font-size: 18px; color: #333;'>
        Powered by the <strong>ReAct Pattern</strong><br>
        Generate structured insights from the web using AI
    </p>
    """,
    unsafe_allow_html=True,
)

# --- Input Section ---
st.markdown("---")
topic = st.text_input("📌 Enter a research topic", placeholder="e.g. The Future of Quantum Computing")

if topic:
    if st.button("🚀 Generate Report"):
        with st.spinner("🧠 Thinking and fetching web results..."):
            agent = ReActAgent(topic)
            questions = agent.generate_questions()

            st.success("✅ Research Questions Generated")
            with st.expander("📋 View Generated Questions", expanded=True):
                for i, q in enumerate(questions, 1):
                    st.markdown(f"**{i}.** {q}")

            st.info("🌐 Searching for accurate answers...")
            answers = agent.search_answers()

            report = generate_report(topic, questions, answers)

            st.markdown("---")
            st.markdown("## 📄 Final Report", unsafe_allow_html=True)
            st.markdown(report, unsafe_allow_html=True)

            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name="research_report.md",
                mime="text/markdown",
                help="Download your research as a Markdown file"
            )
else:
    st.warning("👈 Please enter a topic to get started.")

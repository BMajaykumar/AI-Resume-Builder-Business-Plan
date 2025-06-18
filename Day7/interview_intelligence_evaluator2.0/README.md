# Interview Intelligence Evaluator

A modular, multi-agent AI system for systematically evaluating interview performance based on audio inputs. This tool assesses both technical knowledge and communication skills to generate structured, actionable reports.

---

## 📦 Features

* 🎙️ **Transcription Agent**: Converts uploaded `.wav` interview audio into structured transcripts using Whisper.
* 🗣️ **Soft Skill Detection Agent**: Examines transcripts for behavioral communication markers (hesitation, verbosity, filler words, tone, jargon) using Gemini API.
* 📚 **Knowledge Validation Agent (RAG-Enabled)**: Compares interview responses to reference prep materials via Retrieval-Augmented Generation (RAG) using FAISS and HuggingFace embeddings.
* 📄 **Report Generation Agent**: Compiles structured reports summarizing:

  * Technical Strengths
  * Communication Effectiveness
  * Hiring Readiness

---

## ⚙️ Installation

### 1️⃣ Clone or Download the Repository

```
git clone <repo_url>
```

Or download [Interview\_Intelligence\_Evaluator.zip](link-to-zip)

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt

---

## 🚀 Running the Application

```bash
streamlit run main.py
```

Open browser at: `http://localhost:8501`

---

## 📝 Usage

1. Upload `.wav` audio file of the interview.
2. The system will:

   * Transcribe the audio.
   * Analyze communication skills.
   * Validate answers against reference materials.
   * Generate a final report.

---

## 🔑 Configuration

* **Gemini API Key**:

  * Obtain from [Google AI Studio](https://makersuite.google.com/)
  * Set in `main.py` → `GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"`

* **Reference Materials for Knowledge Validation**:

  * Place `.txt` documents in `data/interview_prep_materials/`

---

## 👨‍💻 Author

**Ajay Kumar**

* GitHub: [https://github.com/BMajaykumar](https://github.com/BMajaykumar)
* LinkedIn: [https://www.linkedin.com/in/bm-ajaykumar/](https://www.linkedin.com/in/bm-ajaykumar/)

---

Built with ❤️ using LangChain, Gemini API, Whisper, and Streamlit.

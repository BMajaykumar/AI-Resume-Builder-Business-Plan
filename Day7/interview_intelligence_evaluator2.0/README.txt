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
```

### 3️⃣ Install and Setup ffmpeg (Required by Whisper)

* Download: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
* Add `/bin` to your system PATH.

---

## 📁 Directory Structure

```
Interview_Intelligence_Evaluator/
├── main.py                     # Main Streamlit App
├── data/
│   └── interview_prep_materials/  # Reference documents for Knowledge Validation (Add .txt files here)
└── README.md
```

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

## 📦 Requirements (Partial)

```txt
streamlit
google-generativeai
openai-whisper
ffmpeg-python
sentence-transformers
faiss-cpu
```

---

## 🖥️ Deployment Ideas

* 📦 Docker (optional)
* ☁️ Streamlit Community Cloud
* 🌐 Custom domain hosting (with authentication for sensitive data)

---

## 🤝 Contributing

PRs are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📜 License

MIT License

---

## 🧩 TODOs & Enhancements

* PDF/Markdown Report Export
* User Authentication (for SaaS)
* More granular soft skill breakdown
* Analytics dashboard for multiple interviews

---

## 👨‍💻 Author

**Ajay Kumar**

* GitHub: [https://github.com/BMajaykumar](https://github.com/BMajaykumar)
* LinkedIn: [https://www.linkedin.com/in/bm-ajaykumar/](https://www.linkedin.com/in/bm-ajaykumar/)

---

Built with ❤️ using LangChain, Gemini API, Whisper, and Streamlit.

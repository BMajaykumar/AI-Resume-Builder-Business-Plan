from config import get_gemini_model

def generate_report(transcript, softskills, knowledge_eval):
    model = get_gemini_model()
    prompt = f"""Generate a structured interview evaluation report with the following:

Transcript:
{transcript}

Soft Skills Analysis:
{softskills}

Knowledge Evaluation:
{knowledge_eval}

Summarize this into:
- Communication Score (0-10)
- Technical Competence (0-10)
- Recommendation (Hire / Consider / No Hire)
- Final Comments
"""
    return model.generate_content(prompt).text

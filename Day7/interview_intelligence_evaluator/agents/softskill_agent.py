from config import get_gemini_model

def analyze_soft_skills(transcript):
    model = get_gemini_model()
    prompt = f"""Analyze this interview transcript for soft skills like:
- hesitation
- filler words
- verbosity
- confidence
- use of jargon

Transcript:
{transcript}
"""
    response = model.generate_content(prompt)
    return response.text

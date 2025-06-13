import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Missing GEMINI_API_KEY in .env file")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

def summarize_text(content):
    prompt = f"Summarize the following educational content into concise bullet points:\n\n{content}"
    response = model.generate_content(prompt)
    return response.text

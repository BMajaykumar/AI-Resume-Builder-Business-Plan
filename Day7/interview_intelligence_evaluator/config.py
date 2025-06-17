import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_gemini_model():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=os.environ.get("GOOGLE_API_KEY")
    )

import os
import google.generativeai as genai
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Tavily client for search
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class ReActAgent:
    def __init__(self, topic):
        self.topic = topic
        self.questions = []
        self.answers = {}

    # 🔹 Generate short & simple questions
    def generate_questions(self):
        prompt = f"Generate 5-6 short and simple questions about the topic: {self.topic}"
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        text = response.text
        self.questions = [line.strip("-• ") for line in text.strip().split("\n") if line.strip()]
        return self.questions

    # 🔹 Get concise web-based answers
    def search_answers(self):
        for question in self.questions:
            trimmed_question = question[:400]  # Tavily limit safeguard
            try:
                result = tavily.search(
                    query=trimmed_question,
                    search_depth="advanced",
                    include_answer=True
                )
                top_results = result.get("results", [])[:2]  # Use 2 sources for brevity

                # Summarize content to max ~200 characters per source
                summary_points = [
                    f"🔹 {r.get('content', 'No content')[:150].strip()}..."
                    for r in top_results
                ]

                summary = "\n".join(summary_points)
                self.answers[question] = summary

            except Exception as e:
                self.answers[question] = f"❌ Error fetching results: {str(e)}"

        return self.answers

import os
import logging
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from typing import Dict, List
from config import GOOGLE_API_KEY, GEMINI_MODEL

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize memory
memory = ConversationBufferMemory()

# Initialize Gemini LLM
try:
    llm = GoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise Exception(f"LLM initialization failed: {e}")

# Prompt template
challenge_template = PromptTemplate(
    input_variables=["clustered_ideas", "theme"],
    template="""Given clustered ideas: {clustered_ideas}, and theme: {theme}, perform empathy-based analysis:
1. Identify specific pain points for each idea.
2. List affected user types and stakeholders.
3. Explain why each problem is important.
4. Filter out shallow ideas (e.g., low human impact or feasibility).
Output for each retained idea:
- Idea: [idea]
  Pain Point: [specific problem]
  Affected Users/Stakeholders: [users]
  Importance: [reason]
Filtered ideas: 'Filtered: [idea] (Reason: [reason])'
If ideas are vague, provide generic analysis or filter them."""
)

def detect_challenges(clustered_ideas: Dict[str, List[Dict[str, List[str]]]], theme: str) -> Dict:
    """Detect challenges for clustered ideas using direct LLM chain."""
    if not clustered_ideas or not any(clustered_ideas.values()):
        logger.error("No valid clustered ideas provided.")
        return {"error": "No valid clustered ideas provided.", "challenges": [], "filtered_ideas": []}

    # Convert to string
    ideas_str = ""
    for theme_name, ideas in clustered_ideas.items():
        ideas_str += f"Theme: {theme_name}\n"
        for item in ideas:
            ideas_str += f"- Idea: {item['idea']}, Tags: {item['tags']}\n"

    logger.info(f"Input to analyze challenges: {ideas_str}")

    # Create chain
    chain = challenge_template | llm

    # Run chain
    try:
        response = chain.invoke({"clustered_ideas": ideas_str, "theme": theme})
        logger.info(f"Challenge response: {response}")
    except Exception as e:
        logger.error(f"Challenge execution error: {e}")
        return {
            "error": f"Error analyzing challenges: {e}",
            "challenges": [],
            "filtered_ideas": [{"idea": idea["idea"], "reason": f"Analysis failed: {e}"}
                              for theme in clustered_ideas.values() for idea in theme]
        }

    # Parse response
    challenges = []
    filtered_ideas = []
    current_idea = None
    current_challenge = {}
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("- Idea:"):
            if current_idea:
                challenges.append(current_challenge)
            current_idea = line.split("Idea:")[1].strip()
            current_challenge = {"idea": current_idea}
        elif line.startswith("Pain Point:") and current_idea:
            current_challenge["pain_point"] = line.replace("Pain Point:", "").strip()
        elif line.startswith("Affected Users/Stakeholders:") and current_idea:
            current_challenge["users_stakeholders"] = line.replace("Affected Users/Stakeholders:", "").strip()
        elif line.startswith("Importance:") and current_idea:
            current_challenge["importance"] = line.replace("Importance:", "").strip()
        elif line.startswith("Filtered:"):
            idea_reason = line.replace("Filtered:", "").strip()
            idea = idea_reason.split("(Reason:")[0].strip()
            reason = idea_reason.split("(Reason:")[1].strip(")")
            filtered_ideas.append({"idea": idea, "reason": reason})

    if current_idea:
        challenges.append(current_challenge)

    # Save to memory
    memory.save_context(
        {"input": f"Analyze challenges for clustered ideas: {ideas_str}, theme: {theme}"},
        {"output": response}
    )

    return {
        "challenges": challenges,
        "filtered_ideas": filtered_ideas
    }

def main():
    """Test the chain."""
    sample_clustered_ideas = {
        "agentic AI": [
            {"idea": "Explore a agentic AI solution using AI", "tags": ["AI", "exploration"]},
            {"idea": "Optimize a process in agentic AI", "tags": ["optimization", "AI"]},
            {"idea": "Address a challenge in Computer Science", "tags": ["challenge", "computer science"]}
        ]
    }
    sample_theme = "agentic AI"
    result = detect_challenges(sample_clustered_ideas, sample_theme)
    print("Challenges:", result["challenges"])
    print("Filtered Ideas:", result["filtered_ideas"])
    print("\nConversation Memory:", memory.load_memory_variables({}))

if __name__ == "__main__":
    main()
import os
import logging
from typing import Dict, List

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import Tool, initialize_agent, AgentType

from config import GOOGLE_API_KEY, GEMINI_MODEL

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Memory ===
memory = ConversationBufferMemory()

# === LLM ===
try:
    llm = GoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise Exception(f"LLM initialization failed: {e}")

# === Prompt Template ===
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

# === Challenge Analyzer Function ===
def detect_challenges(clustered_ideas: Dict[str, List[Dict[str, List[str]]]], theme: str) -> Dict:
    if not clustered_ideas or not any(clustered_ideas.values()):
        logger.error("No valid clustered ideas provided.")
        return {"error": "No valid clustered ideas provided.", "challenges": [], "filtered_ideas": []}

    ideas_str = ""
    for theme_name, ideas in clustered_ideas.items():
        ideas_str += f"Theme: {theme_name}\n"
        for item in ideas:
            ideas_str += f"- Idea: {item['idea']}, Tags: {item['tags']}\n"

    logger.info(f"Input to analyze challenges: {ideas_str}")

    chain = challenge_template | llm

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

    # === Parse Output ===
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

    memory.save_context(
        {"input": f"Analyze challenges for clustered ideas: {ideas_str}, theme: {theme}"},
        {"output": response}
    )

    return {
        "challenges": challenges,
        "filtered_ideas": filtered_ideas
    }

# === Tool Wrapper ===
def detect_challenges_tool(input_text: str) -> str:
    """
    Input format:
    theme: X | idea: A, tags: [tag1, tag2] | idea: B, tags: [tag3]
    """
    try:
        parts = input_text.strip().split("|")
        theme = parts[0].replace("theme:", "").strip()
        clustered_ideas = {"from_input": []}
        for p in parts[1:]:
            idea_part, tags_part = p.strip().split(", tags:")
            idea = idea_part.replace("idea:", "").strip()
            tags = tags_part.strip().replace("[", "").replace("]", "").split(",")
            clustered_ideas["from_input"].append({"idea": idea, "tags": [tag.strip() for tag in tags]})
        result = detect_challenges(clustered_ideas, theme)
        output = "🧠 Challenges:\n"
        for ch in result["challenges"]:
            output += f"- {ch['idea']} (Pain: {ch['pain_point']}, Users: {ch['users_stakeholders']}, Why: {ch['importance']})\n"
        output += "\n❌ Filtered:\n"
        for f in result["filtered_ideas"]:
            output += f"- {f['idea']} (Reason: {f['reason']})\n"
        return output
    except Exception as e:
        return f"Tool Error: {e}"

# === Define the Tool ===
challenge_tool = Tool(
    name="ChallengeDetectorTool",
    func=detect_challenges_tool,
    description=(
        "Analyzes clustered project ideas to detect challenges. "
        "Input format: 'theme: X | idea: ..., tags: [a, b] | idea: ..., tags: [c]'"
    )
)

# === Agent ===
agent = initialize_agent(
    tools=[challenge_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# === Main ===
def main():
    print("🔍 Challenge Detector Agent")

    query = (
        "theme: agentic AI | "
        "idea: Explore agentic AI solution using AI, tags: [AI, exploration] | "
        "idea: Optimize a process in agentic AI, tags: [optimization, AI] | "
        "idea: Address a challenge in Computer Science, tags: [generic, challenge]"
    )

    print("\n🤖 Agent Response:")
    result = agent.run(query)
    print(result)

    print("\n🧠 Memory:")
    print(memory.load_memory_variables({}))


if __name__ == "__main__":
    main()

import os
import logging
from typing import Dict, List, Optional

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
scoring_template = PromptTemplate(
    input_variables=["challenges", "theme", "survey_results"],
    template="""Given challenge-mapped ideas: {challenges}, and theme: {theme}, score each idea based on:
1. Feasibility (40%): Technical and resource feasibility (0-100).
2. Impact (30%): Potential to address pain points and benefit stakeholders (0-100).
3. Empathy (30%): Depth of user-centered focus and stakeholder alignment (0-100).
Survey results (if provided): {survey_results}

For each idea, provide:
- Idea: [idea]
  Feasibility: [score]/100 (Reason: [reason])
  Impact: [score]/100 (Reason: [reason])
  Empathy: [score]/100 (Reason: [reason])
  Total Score: [weighted score]/100
If survey results are provided, adjust scores based on user feedback (e.g., increase/decrease by up to 20 points per criterion).
Output only the top 3 ideas, ranked by total score, with a scoring breakdown.
If challenges are vague or invalid, provide low scores and explain.
Format:
- Rank 1: [idea]
  Feasibility: [score]/100 (Reason: [reason])
  Impact: [score]/100 (Reason: [reason])
  Empathy: [score]/100 (Reason: [reason])
  Total Score: [score]/100
- Rank 2: ...
"""
)

# === Main Logic ===
def score_opportunities(challenges: List[Dict], theme: str, survey_results: Optional[str] = None) -> Dict:
    if not challenges:
        logger.error("No valid challenges provided.")
        return {"error": "No valid challenges provided.", "ranked_opportunities": []}

    challenges_str = ""
    for challenge in challenges:
        challenges_str += f"- Idea: {challenge['idea']}\n"
        challenges_str += f"  Pain Point: {challenge.get('pain_point', 'N/A')}\n"
        challenges_str += f"  Affected Users/Stakeholders: {challenge.get('users_stakeholders', 'N/A')}\n"
        challenges_str += f"  Importance: {challenge.get('importance', 'N/A')}\n"

    if not survey_results:
        survey_results = "\n".join(
            [f"Idea: {challenge['idea']}, User Rating: {80 if 'Optimize' in challenge['idea'] else 50}/100"
             for challenge in challenges]
        )
        logger.info("No survey results provided, using mock survey results.")

    logger.info(f"Input to score opportunities:\n{challenges_str}\nSurvey Results:\n{survey_results}")

    chain = scoring_template | llm

    try:
        response = chain.invoke({"challenges": challenges_str, "theme": theme, "survey_results": survey_results})
        logger.info(f"Scoring response: {response}")
    except Exception as e:
        logger.error(f"Error scoring opportunities: {e}")
        return {
            "error": f"Error scoring opportunities: {e}",
            "ranked_opportunities": []
        }

    # === Parse Response ===
    ranked_opportunities = []
    current_opportunity = {}
    current_rank = None
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("- Rank"):
            if current_rank and current_opportunity:
                ranked_opportunities.append(current_opportunity)
            current_rank = line.split(":")[0].replace("- ", "").strip()
            current_opportunity = {"rank": current_rank, "idea": line.split(":")[1].strip()}
        elif line.startswith("Feasibility:") and current_rank:
            score, reason = line.replace("Feasibility:", "").split("(Reason:")
            current_opportunity["feasibility"] = int(score.split("/")[0].strip())
            current_opportunity["feasibility_reason"] = reason.strip(")")
        elif line.startswith("Impact:") and current_rank:
            score, reason = line.replace("Impact:", "").split("(Reason:")
            current_opportunity["impact"] = int(score.split("/")[0].strip())
            current_opportunity["impact_reason"] = reason.strip(")")
        elif line.startswith("Empathy:") and current_rank:
            score, reason = line.replace("Empathy:", "").split("(Reason:")
            current_opportunity["empathy"] = int(score.split("/")[0].strip())
            current_opportunity["empathy_reason"] = reason.strip(")")
        elif line.startswith("Total Score:") and current_rank:
            current_opportunity["total_score"] = float(line.replace("Total Score:", "").split("/")[0].strip())

    if current_rank and current_opportunity:
        ranked_opportunities.append(current_opportunity)

    memory.save_context(
        {"input": f"Score opportunities for challenges: {challenges_str}, theme: {theme}, survey: {survey_results}"},
        {"output": response}
    )

    return {
        "ranked_opportunities": ranked_opportunities[:3]
    }

# === Tool Wrapper ===
def score_opportunities_tool(input_text: str) -> str:
    """
    Input format:
    idea1:..., pain_point:..., users:..., importance:...|idea2:..., ...
    Example: 'theme: agentic AI | idea: X, pain_point: Y, users: Z, importance: A | idea: B, ...'
    """
    try:
        sections = input_text.strip().split("|")
        theme = sections[0].replace("theme:", "").strip()
        challenges = []
        for idea_line in sections[1:]:
            parts = {kv.split(":")[0].strip(): kv.split(":")[1].strip() for kv in idea_line.split(",")}
            challenges.append({
                "idea": parts.get("idea", ""),
                "pain_point": parts.get("pain_point", ""),
                "users_stakeholders": parts.get("users", ""),
                "importance": parts.get("importance", "")
            })
        result = score_opportunities(challenges, theme)
        return "\n\n".join([f"{r['rank']} - {r['idea']}\nScore: {r['total_score']}/100" for r in result["ranked_opportunities"]])
    except Exception as e:
        return f"Tool error: {e}"

# === Define the Tool ===
scorer_tool = Tool(
    name="OpportunityScorerTool",
    func=score_opportunities_tool,
    description=(
        "Scores and ranks project ideas based on feasibility, impact, empathy. "
        "Format: 'theme: X | idea: ..., pain_point: ..., users: ..., importance: ... | idea: ...'"
    )
)

# === Agent ===
agent = initialize_agent(
    tools=[scorer_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# === MAIN ===
def main():
    print("🎯 Opportunity Scorer Agent")

    query = (
        "theme: agentic AI | "
        "idea: Optimize a process in agentic AI, pain_point: High cost, users: Developers, importance: Reduce expense | "
        "idea: Explore agentic AI using AI, pain_point: Framework inaccessibility, users: Researchers, importance: Improve tools | "
        "idea: Address a challenge in CS, pain_point: N/A, users: N/A, importance: N/A"
    )

    print("\n🤖 Agent Output:")
    result = agent.run(query)
    print(result)

    print("\n🧠 Memory:")
    print(memory.load_memory_variables({}))


if __name__ == "__main__":
    main()

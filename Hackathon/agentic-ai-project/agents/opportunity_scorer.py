import os
import logging
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from typing import Dict, List, Optional
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

# Prompt template for scoring opportunities
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
- Rank 2: [idea]
  ...
"""
)

def score_opportunities(challenges: List[Dict], theme: str, survey_results: Optional[str] = None) -> Dict:
    """Score and rank challenge-mapped ideas."""
    if not challenges:
        logger.error("No valid challenges provided.")
        return {"error": "No valid challenges provided.", "ranked_opportunities": []}

    # Convert challenges to string
    challenges_str = ""
    for challenge in challenges:
        challenges_str += f"- Idea: {challenge['idea']}\n"
        challenges_str += f"  Pain Point: {challenge.get('pain_point', 'N/A')}\n"
        challenges_str += f"  Affected Users/Stakeholders: {challenge.get('users_stakeholders', 'N/A')}\n"
        challenges_str += f"  Importance: {challenge.get('importance', 'N/A')}\n"

    # Use mock survey results if none provided
    if not survey_results:
        survey_results = "\n".join(
            [f"Idea: {challenge['idea']}, User Rating: {80 if 'Optimize' in challenge['idea'] else 50}/100"
             for challenge in challenges]
        )
        logger.info("No survey results provided, using mock survey results.")

    logger.info(f"Input to score opportunities: {challenges_str}\nSurvey Results: {survey_results}")

    # Create chain
    chain = scoring_template | llm

    # Run chain
    try:
        response = chain.invoke({"challenges": challenges_str, "theme": theme, "survey_results": survey_results})
        logger.info(f"Scoring response: {response}")
    except Exception as e:
        logger.error(f"Error scoring opportunities: {e}")
        return {
            "error": f"Error scoring opportunities: {e}",
            "ranked_opportunities": []
        }

    # Parse response
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

    # Save to memory
    memory.save_context(
        {"input": f"Score opportunities for challenges: {challenges_str}, theme: {theme}, survey: {survey_results}"},
        {"output": response}
    )

    return {
        "ranked_opportunities": ranked_opportunities[:3]  # Ensure only top 3
    }

def main():
    """Test the opportunity scorer."""
    sample_challenges = [
        {
            "idea": "Optimize a process in agentic AI",
            "pain_point": "High computational costs in agentic AI processes",
            "users_stakeholders": "Businesses, developers",
            "importance": "Reducing costs improves scalability."
        },
        {
            "idea": "Explore a agentic AI solution using AI",
            "pain_point": "Lack of accessible frameworks",
            "users_stakeholders": "Developers, researchers",
            "importance": "Simplifies development."
        },
        {
            "idea": "Address a challenge in Computer Science",
            "pain_point": "N/A",
            "users_stakeholders": "N/A",
            "importance": "N/A"
        }
    ]
    sample_theme = "agentic AI"
    sample_survey_results = """Idea: Optimize a process in agentic AI, User Rating: 85/100
Idea: Explore a agentic AI solution using AI, User Rating: 60/100
Idea: Address a challenge in Computer Science, User Rating: 40/100"""
    
    result = score_opportunities(sample_challenges, sample_theme, sample_survey_results)
    print("Ranked Opportunities:", result["ranked_opportunities"])
    print("\nConversation Memory:", memory.load_memory_variables({}))

if __name__ == "__main__":
    main()
import os
import logging
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from typing import Dict, Optional
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

# Prompt template for refining problem statement
refiner_template = PromptTemplate(
    input_variables=["opportunity", "theme"],
    template="""Given the top-ranked opportunity: {opportunity}, and theme: {theme}, craft a refined problem statement and knowledge plan:
1. Problem Statement (1–2 sentences):
   - Include who (stakeholders/users), what (specific problem), and why (importance).
   - Ensure clarity and academic tone.
2. Knowledge Plan:
   - What we know: List current understanding based on the opportunity.
   - What we need to learn: List knowledge gaps to address for solution development.
Output format:
Problem Statement: [1–2 sentences]
Knowledge Plan:
- What we know:
  - [point]
  - [point]
- What we need to learn:
  - [point]
  - [point]
If the opportunity is vague, provide a generic but reasonable output."""
)

def refine_problem_statement(opportunity: Dict, theme: str) -> Dict:
    """Refine the top-ranked opportunity into a problem statement and knowledge plan."""
    if not opportunity or not opportunity.get("idea"):
        logger.error("No valid opportunity provided.")
        return {"error": "No valid opportunity provided.", "problem_statement": "", "knowledge_plan": {}}

    # Convert opportunity to string
    opportunity_str = f"Idea: {opportunity['idea']}\n"
    opportunity_str += f"Feasibility: {opportunity.get('feasibility', 'N/A')}/100 (Reason: {opportunity.get('feasibility_reason', 'N/A')})\n"
    opportunity_str += f"Impact: {opportunity.get('impact', 'N/A')}/100 (Reason: {opportunity.get('impact_reason', 'N/A')})\n"
    opportunity_str += f"Empathy: {opportunity.get('empathy', 'N/A')}/100 (Reason: {opportunity.get('empathy_reason', 'N/A')})\n"
    opportunity_str += f"Total Score: {opportunity.get('total_score', 'N/A')}/100"

    logger.info(f"Input to refine problem statement: {opportunity_str}")

    # Create chain
    chain = refiner_template | llm

    # Run chain
    try:
        response = chain.invoke({"opportunity": opportunity_str, "theme": theme})
        logger.info(f"Refiner response: {response}")
    except Exception as e:
        logger.error(f"Error refining problem statement: {e}")
        return {
            "error": f"Error refining problem statement: {e}",
            "problem_statement": "",
            "knowledge_plan": {}
        }

    # Parse response
    problem_statement = ""
    knowledge_plan = {"what_we_know": [], "what_we_need_to_learn": []}
    current_section = None
    for line in response.strip().split("\n"):
        line = line.strip()
        if line.startswith("Problem Statement:"):
            problem_statement = line.replace("Problem Statement:", "").strip()
        elif line.startswith("- What we know:"):
            current_section = "what_we_know"
        elif line.startswith("- What we need to learn:"):
            current_section = "what_we_need_to_learn"
        elif line.startswith("- ") and current_section:
            knowledge_plan[current_section].append(line.replace("- ", "").strip())

    # Save to memory
    memory.save_context(
        {"input": f"Refine problem statement for opportunity: {opportunity_str}, theme: {theme}"},
        {"output": response}
    )

    return {
        "problem_statement": problem_statement,
        "knowledge_plan": knowledge_plan
    }

def main():
    """Test the problem refiner."""
    sample_opportunity = {
        "rank": "Rank 1",
        "idea": "Optimize a process in agentic AI",
        "feasibility": 85,
        "feasibility_reason": "Technically feasible with existing AI frameworks",
        "impact": 90,
        "impact_reason": "Reduces costs for businesses",
        "empathy": 80,
        "empathy_reason": "Addresses developer and business needs",
        "total_score": 85.5
    }
    sample_theme = "agentic AI"
    result = refine_problem_statement(sample_opportunity, sample_theme)
    print("Problem Statement:", result["problem_statement"])
    print("Knowledge Plan:", result["knowledge_plan"])
    print("\nConversation Memory:", memory.load_memory_variables({}))

if __name__ == "__main__":
    main()
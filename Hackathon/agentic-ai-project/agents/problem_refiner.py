import os
import logging
from typing import Dict

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType

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

# === Function to Refine Problem Statement ===
def refine_problem_statement(opportunity: Dict, theme: str) -> Dict:
    if not opportunity or not opportunity.get("idea"):
        logger.error("No valid opportunity provided.")
        return {"error": "No valid opportunity provided.", "problem_statement": "", "knowledge_plan": {}}

    opportunity_str = f"Idea: {opportunity['idea']}\n"
    opportunity_str += f"Feasibility: {opportunity.get('feasibility', 'N/A')}/100 (Reason: {opportunity.get('feasibility_reason', 'N/A')})\n"
    opportunity_str += f"Impact: {opportunity.get('impact', 'N/A')}/100 (Reason: {opportunity.get('impact_reason', 'N/A')})\n"
    opportunity_str += f"Empathy: {opportunity.get('empathy', 'N/A')}/100 (Reason: {opportunity.get('empathy_reason', 'N/A')})\n"
    opportunity_str += f"Total Score: {opportunity.get('total_score', 'N/A')}/100"

    logger.info(f"Input to refine problem statement: {opportunity_str}")

    chain = refiner_template | llm

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

    memory.save_context(
        {"input": f"Refine problem statement for opportunity: {opportunity_str}, theme: {theme}"},
        {"output": response}
    )

    return {
        "problem_statement": problem_statement,
        "knowledge_plan": knowledge_plan
    }

# === Tool Function Wrapper ===
def refine_tool_func(input_text: str) -> str:
    """
    Input format:
    theme: ..., idea: ..., feasibility: ..., feasibility_reason: ..., impact: ..., impact_reason: ..., empathy: ..., empathy_reason: ..., total_score: ...
    """
    try:
        parts = {kv.split(":")[0].strip(): kv.split(":")[1].strip() for kv in input_text.split(",")}
        opportunity = {
            "idea": parts.get("idea", ""),
            "feasibility": int(parts.get("feasibility", 80)),
            "feasibility_reason": parts.get("feasibility_reason", ""),
            "impact": int(parts.get("impact", 80)),
            "impact_reason": parts.get("impact_reason", ""),
            "empathy": int(parts.get("empathy", 80)),
            "empathy_reason": parts.get("empathy_reason", ""),
            "total_score": float(parts.get("total_score", 80.0))
        }
        theme = parts.get("theme", "general")
        result = refine_problem_statement(opportunity, theme)
        output = f"📌 Problem Statement:\n{result['problem_statement']}\n\n📚 Knowledge Plan:\n"
        output += "What we know:\n" + "\n".join(f"- {item}" for item in result["knowledge_plan"]["what_we_know"]) + "\n"
        output += "What we need to learn:\n" + "\n".join(f"- {item}" for item in result["knowledge_plan"]["what_we_need_to_learn"])
        return output
    except Exception as e:
        return f"Tool error: {e}"

# === Define the Tool ===
refiner_tool = Tool(
    name="ProblemStatementRefiner",
    func=refine_tool_func,
    description=(
        "Refines a top-ranked opportunity into a problem statement and knowledge plan. "
        "Input format must be: 'theme: ..., idea: ..., feasibility: ..., feasibility_reason: ..., impact: ..., impact_reason: ..., empathy: ..., empathy_reason: ..., total_score: ...'"
    )
)

# === Initialize AgentExecutor ===
agent_executor = initialize_agent(
    tools=[refiner_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# === MAIN ===
def main():
    print("🎯 Testing Agentic Problem Refiner Agent")

    # Sample input
    query = (
        "theme: agentic AI, idea: Optimize a process in agentic AI, "
        "feasibility: 85, feasibility_reason: Technically feasible with existing AI frameworks, "
        "impact: 90, impact_reason: Reduces costs for businesses, "
        "empathy: 80, empathy_reason: Addresses developer and business needs, "
        "total_score: 85.5"
    )

    # Agent Execution
    print("\n🤖 Agent Response:")
    result = agent_executor.run(query)
    print(result)

    # Show memory
    print("\n🧠 Memory:")
    print(memory.load_memory_variables({}))


if __name__ == "__main__":
    main()

import os
import logging
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.agents import AgentExecutor
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
    raise Exception(f"LLM initialization failed. Check GOOGLE_API_KEY and GEMINI_MODEL: {e}")

# Prompt templates
cluster_tag_template = PromptTemplate(
    input_variables=["ideas", "theme"],
    template="""Given the ideas: {ideas}, and theme: {theme}, categorize into themes (e.g., agentic AI, automation) and apply tags (e.g., AI, optimization). If ideas are vague, use generic themes and tags. Output:
Theme: [theme]
- Idea: [idea], Tags: [tag1, tag2, ...]
Ensure one theme aligns with {theme}."""
)

diagram_template = PromptTemplate(
    input_variables=["ideas", "theme"],
    template="""For ideas: {ideas}, and theme: {theme}, suggest text-based diagrams or use-case flows (2-3 sentences each). If ideas are vague, provide generic suggestions. Output:
Suggestion for [idea]: [description]"""
)

def cluster_and_tag_ideas(ideas: str, theme: str) -> str:
    """Tool to cluster and tag ideas."""
    chain = cluster_tag_template | llm
    try:
        response = chain.invoke({"ideas": ideas, "theme": theme})
        logger.info(f"Cluster response: {response}")
        if not response.strip() or "Theme:" not in response:
            logger.warning("Invalid cluster response, using fallback.")
            return f"Theme: {theme}\n" + "\n".join(
                [f"- Idea: {idea}, Tags: [generic, {theme.lower()}]" for idea in ideas.split("\n") if idea.strip()]
            )
        return response.strip()
    except Exception as e:
        logger.error(f"Error clustering: {e}")
        return f"Theme: {theme}\n" + "\n".join(
            [f"- Idea: {idea}, Tags: [error, {theme.lower()}]" for idea in ideas.split("\n") if idea.strip()]
        )

def suggest_diagrams_tool(ideas: str, theme: str) -> str:
    """Tool to suggest diagrams."""
    chain = diagram_template | llm
    try:
        response = chain.invoke({"ideas": ideas, "theme": theme})
        logger.info(f"Diagram response: {response}")
        if not response.strip() or "Suggestion for" not in response:
            logger.warning("Invalid diagram response, using fallback.")
            return "\n".join(
                [f"Suggestion for {idea}: A generic flowchart showing the implementation steps for a {theme} solution." 
                 for idea in ideas.split("\n") if idea.strip()]
            )
        return response.strip()
    except Exception as e:
        logger.error(f"Error suggesting diagrams: {e}")
        return "\n".join(
            [f"Suggestion for {idea}: A generic flowchart showing the implementation steps for a {theme} solution." 
             for idea in ideas.split("\n") if idea.strip()]
        )

# Define tools
tools = [
    Tool(
        name="ClusterAndTagIdeas",
        func=lambda input_str: cluster_and_tag_ideas(input_str.split("|")[0], input_str.split("|")[1]),
        description="Clusters ideas into themes and applies tags. Input: ideas|theme"
    ),
    Tool(
        name="SuggestDiagrams",
        func=lambda input_str: suggest_diagrams_tool(input_str.split("|")[0], input_str.split("|")[1]),
        description="Suggests text-based diagrams. Input: ideas|theme"
    )
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# Create AgentExecutor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    handle_parsing_errors=True
)

def process_ideas(ideas: List[str], theme: str) -> Dict:
    """Process ideas, cluster, tag, and suggest diagrams."""
    if not ideas:
        return {"error": "No ideas provided.", "clustered_ideas": {}, "diagram_suggestions": []}

    ideas_input = "\n".join(ideas)
    input_str = f"{ideas_input}|{theme}"

    # Cluster and tag
    try:
        cluster_response = agent_executor.run(
            f"Cluster and tag ideas: {ideas_input}, theme: {theme}"
        )
    except Exception as e:
        logger.error(f"Cluster execution error: {e}")
        cluster_response = f"Theme: {theme}\n" + "\n".join(
            [f"- Idea: {idea}, Tags: [generic, {theme.lower()}]" for idea in ideas]
        )

    # Parse clusters
    clusters = {}
    current_theme = None
    for line in cluster_response.strip().split("\n"):
        line = line.strip()
        if line.startswith("Theme:"):
            current_theme = line.replace("Theme:", "").strip()
            clusters[current_theme] = []
        elif line.startswith("- Idea:") and current_theme:
            try:
                idea_part = line.split(", Tags:")[0].replace("- Idea:", "").strip()
                tags_part = line.split(", Tags:")[1].strip().strip("[]").split(", ")
                clusters[current_theme].append({"idea": idea_part, "tags": tags_part})
            except IndexError:
                logger.warning(f"Failed to parse line: {line}")
                clusters[current_theme].append({"idea": line.replace("- Idea:", "").strip(), "tags": ["parse_error"]})

    if not clusters:
        logger.warning("No clusters parsed, using fallback.")
        clusters[theme] = [{"idea": idea, "tags": ["generic", theme.lower()]} for idea in ideas]

    # Suggest diagrams
    try:
        diagram_response = agent_executor.run(
            f"Suggest diagrams for ideas: {ideas_input}, theme: {theme}"
        )
    except Exception as e:
        logger.error(f"Diagram execution error: {e}")
        diagram_response = "\n".join(
            [f"Suggestion for {idea}: A generic flowchart showing the implementation steps for a {theme} solution." 
             for idea in ideas]
        )

    diagram_suggestions = [
        line.strip() for line in diagram_response.strip().split("\n") if line.strip().startswith("Suggestion for")
    ]
    if not diagram_suggestions:
        logger.warning("No diagram suggestions parsed, using fallback.")
        diagram_suggestions = [
            f"Suggestion for {idea}: A generic flowchart showing the implementation steps for a {theme} solution." 
            for idea in ideas
        ]

    # Save to memory
    memory.save_context(
        {"input": f"Process ideas: {ideas}, theme: {theme}"},
        {"output": f"Clustered: {clusters}\nDiagrams: {diagram_suggestions}"}
    )

    return {
        "clustered_ideas": clusters,
        "diagram_suggestions": diagram_suggestions
    }

def main():
    """Test the agent."""
    sample_ideas = [
        "Explore a agentic AI solution using AI",
        "Optimize a process in agentic AI",
        "Address a challenge in Computer Science"
    ]
    sample_theme = "agentic AI"
    result = process_ideas(sample_ideas, sample_theme)
    print("Clustered Ideas:", result["clustered_ideas"])
    print("Diagram Suggestions:", result["diagram_suggestions"])
    print("\nConversation Memory:", memory.load_memory_variables({}))

if __name__ == "__main__":
    main()
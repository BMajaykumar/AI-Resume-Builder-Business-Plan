import os
from typing import Dict, List
from uuid import uuid4
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI

from config import GOOGLE_API_KEY, VECTORSTORE_PATH, EMBEDDING_MODEL, GEMINI_MODEL

# === Initialize ===
memory = ConversationBufferMemory()
llm = GoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


# === Load FAISS ===
def load_vectorstore():
    if not os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss")):
        raise Exception(
            f"FAISS vector store not found at {VECTORSTORE_PATH}. "
            "Please run create_faiss_index.py to generate the index."
        )
    try:
        return FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise Exception(f"Failed to load FAISS vector store: {e}")


vectorstore = load_vectorstore()

# === Prompt Template ===
prompt_template = PromptTemplate(
    input_variables=["metadata", "retrieved_docs"],
    template="""Based on the project metadata {metadata} and these examples from past projects {retrieved_docs}, 
    generate 3-5 brainstorming prompts for a student project. Each prompt must be specific to the domain, 
    actionable, and include a citation to the source document (e.g., document page or title). 
    Format each prompt as: 'Prompt: [prompt text] (Source: [citation])'."""
)


# === Utilities ===
def retrieve_relevant_docs(theme: str, k: int = 5) -> List[str]:
    try:
        docs = vectorstore.similarity_search(theme, k=k)
        return [f"Document {i+1} (Page {doc.metadata.get('page', 'unknown')}): {doc.page_content[:200]}..." 
                for i, doc in enumerate(docs)]
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []


def get_project_metadata(title: str, theme: str, department: str) -> Dict[str, str]:
    return {
        "title": title,
        "theme": theme,
        "department": department,
        "id": str(uuid4())
    }


def generate_prompts(metadata: Dict[str, str]) -> List[str]:
    retrieved_docs = retrieve_relevant_docs(metadata.get("theme", ""))
    chain = prompt_template | llm

    try:
        response = chain.invoke({
            "metadata": str(metadata),
            "retrieved_docs": "\n".join(retrieved_docs) if retrieved_docs else "No relevant documents found."
        })
    except Exception as e:
        print(f"Error generating prompts: {e}")
        return ["Error: Could not generate prompts."]

    prompts = [p.strip() for p in response.strip().split("\n") if p.strip().startswith("Prompt:")]

    if len(prompts) < 3:
        fallback_prompts = [
            f"Prompt: Explore a {metadata.get('theme', 'general')} solution using AI (Source: Generic)",
            f"Prompt: Optimize a process in {metadata.get('theme', 'general')} (Source: Generic)",
            f"Prompt: Address a challenge in {metadata.get('department', 'general')} (Source: Generic)"
        ]
        prompts.extend(fallback_prompts[:5 - len(prompts)])

    prompts = prompts[:5]

    memory.save_context(
        {"input": f"Generate prompts for {metadata}"},
        {"output": "\n".join(prompts)}
    )

    return prompts


# === LangChain Tool Wrapper ===
def generate_prompts_tool_fn(input_text: str) -> str:
    """
    Tool function that parses metadata from input_text and returns brainstorming prompts.
    Format: 'title:..., theme:..., department:...'
    """
    try:
        parts = {kv.split(":")[0].strip(): kv.split(":")[1].strip()
                 for kv in input_text.split(",")}
        metadata = get_project_metadata(
            title=parts.get("title", ""),
            theme=parts.get("theme", ""),
            department=parts.get("department", "")
        )
        prompts = generate_prompts(metadata)
        return "\n".join(prompts)
    except Exception as e:
        return f"Error parsing input or generating prompts: {e}"


# === Register the Tool ===
generate_prompt_tool = Tool(
    name="BrainstormPromptTool",
    func=generate_prompts_tool_fn,
    description=(
        "Use this tool to generate 3-5 brainstorming prompts for a student project. "
        "Input must be: 'title:..., theme:..., department:...'"
    )
)

# === Create Agent ===
agent = initialize_agent(
    tools=[generate_prompt_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)


# === MAIN ===
def main():
    print("🎯 Agentic Prompt Generator Agent")

    # You can change this input to test different queries
    user_query = "title: AI for Mental Health, theme: psychology, department: Computer Science"

    print("\n🤖 Agent Output:")
    result = agent.run(user_query)
    print(result)

    print("\n🧠 Conversation Memory:")
    print(memory.load_memory_variables({}))


if __name__ == "__main__":
    main()

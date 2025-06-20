import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from typing import Dict, List
from uuid import uuid4
from config import GOOGLE_API_KEY, VECTORSTORE_PATH, EMBEDDING_MODEL, GEMINI_MODEL

# Initialize memory for conversation context
memory = ConversationBufferMemory()

# Initialize Gemini LLM
llm = GoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)

# Initialize embeddings for RAG
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# Load FAISS vector store
def load_vectorstore():
    """
    Load the FAISS vector store from the specified path.
    
    Returns:
        FAISS: Loaded vector store
    Raises:
        Exception: If the vector store file does not exist
    """
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

# Prompt template for generating brainstorming prompts
prompt_template = PromptTemplate(
    input_variables=["metadata", "retrieved_docs"],
    template="""Based on the project metadata {metadata} and these examples from past projects {retrieved_docs}, 
    generate 3-5 brainstorming prompts for a student project. Each prompt must be specific to the domain, 
    actionable, and include a citation to the source document (e.g., document page or title). 
    Format each prompt as: 'Prompt: [prompt text] (Source: [citation])'."""
)

def retrieve_relevant_docs(theme: str, k: int = 5) -> List[str]:
    """
    Retrieve relevant documents from FAISS vector store based on project theme.
    
    Args:
        theme (str): Project theme (e.g., "healthcare")
        k (int): Number of documents to retrieve
    
    Returns:
        List[str]: List of document contents with metadata
    """
    try:
        docs = vectorstore.similarity_search(theme, k=k)
        return [f"Document {i+1} (Page {doc.metadata.get('page', 'unknown')}): {doc.page_content[:200]}..." 
                for i, doc in enumerate(docs)]
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []

def generate_prompts(metadata: Dict[str, str]) -> List[str]:
    """
    Generate 3-5 tailored brainstorming prompts using RAG and LLM.
    
    Args:
        metadata (Dict[str, str]): Project metadata (title, theme, department)
    
    Returns:
        List[str]: List of formatted prompts with citations
    """
    # Retrieve relevant documents based on theme
    retrieved_docs = retrieve_relevant_docs(metadata.get("theme", ""))
    
    # Prepare input for LLM
    chain = prompt_template | llm
    try:
        response = chain.invoke({
            "metadata": str(metadata),
            "retrieved_docs": "\n".join(retrieved_docs) if retrieved_docs else "No relevant documents found."
        })
    except Exception as e:
        print(f"Error generating prompts: {e}")
        return ["Error: Could not generate prompts."]
    
    # Parse response into list of prompts
    prompts = [p.strip() for p in response.strip().split("\n") if p.strip().startswith("Prompt:")]
    
    # Ensure 3-5 prompts (fallback to generic prompts if needed)
    if len(prompts) < 3:
        fallback_prompts = [
            f"Prompt: Explore a {metadata.get('theme', 'general')} solution using AI (Source: Generic)",
            f"Prompt: Optimize a process in {metadata.get('theme', 'general')} (Source: Generic)",
            f"Prompt: Address a challenge in {metadata.get('department', 'general')} (Source: Generic)"
        ]
        prompts.extend(fallback_prompts[:5 - len(prompts)])
    
    prompts = prompts[:5]  # Cap at 5 prompts
    
    # Save to memory
    memory.save_context(
        {"input": f"Generate prompts for {metadata}"},
        {"output": "\n".join(prompts)}
    )
    
    return prompts

def get_project_metadata(title: str, theme: str, department: str) -> Dict[str, str]:
    """
    Format project metadata.
    
    Args:
        title (str): Project title
        theme (str): Project theme
        department (str): Department name
    
    Returns:
        Dict[str, str]: Formatted metadata
    """
    return {
        "title": title,
        "theme": theme,
        "department": department,
        "id": str(uuid4())  # Unique ID for tracking
    }

def main():
    """
    Main function for testing the Brainstorm Prompt Generator Agent.
    """
    # Sample metadata for testing
    sample_metadata = get_project_metadata(
        title="AI Healthcare Solution",
        theme="healthcare",
        department="Computer Science"
    )
    
    # Generate prompts
    prompts = generate_prompts(sample_metadata)
    
    # Print results
    print("Generated Brainstorming Prompts:")
    for i, prompt in enumerate(prompts, 1):
        print(f"{i}. {prompt}")
    
    # Print memory for debugging
    print("\nConversation Memory:")
    print(memory.load_memory_variables({}))

if __name__ == "__main__":
    main()
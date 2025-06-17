from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from agents.transcription_agent import transcribe_audio
from agents.softskill_agent import analyze_soft_skills
from agents.knowledge_agent import create_vector_store, validate_knowledge
from agents.report_agent import generate_report

# ✅ Define the shared state
class GraphState(TypedDict):
    audio_path: Optional[str]
    transcript: Optional[str]
    softskills: Optional[str]
    knowledge_eval: Optional[str]
    final_report: Optional[str]


# ✅ Agent 1: Transcription Node
def transcription_node(state: GraphState) -> dict:
    audio_path = state.get("audio_path")
    transcript = transcribe_audio(audio_path)
    return {"transcript": transcript}


# ✅ Agent 2: Soft Skill Analysis Node
def softskill_node(state: GraphState) -> dict:
    transcript = state.get("transcript")
    softskills = analyze_soft_skills(transcript)
    return {"softskills": softskills}


# ✅ Agent 3: Knowledge Validation Node (RAG)
def knowledge_node(state: GraphState) -> dict:
    transcript = state.get("transcript")
    vectorstore = create_vector_store()
    knowledge_eval = validate_knowledge(transcript, vectorstore)
    return {"knowledge_eval": knowledge_eval}


# ✅ Agent 4: Report Generation Node
def report_node(state: GraphState) -> dict:
    transcript = state.get("transcript")
    softskills = state.get("softskills")
    knowledge_eval = state.get("knowledge_eval")
    final_report = generate_report(transcript, softskills, knowledge_eval)
    return {"final_report": final_report}


# ✅ Build LangGraph agent flow
def build_agent_graph():
    builder = StateGraph(GraphState)

    builder.add_node("transcription", transcription_node)
    builder.add_node("analyze_softskills", softskill_node)  # renamed to avoid conflict
    builder.add_node("knowledge_validation", knowledge_node)
    builder.add_node("report", report_node)

    builder.set_entry_point("transcription")
    builder.add_edge("transcription", "analyze_softskills")
    builder.add_edge("analyze_softskills", "knowledge_validation")
    builder.add_edge("knowledge_validation", "report")
    builder.set_finish_point("report")

    return builder.compile()

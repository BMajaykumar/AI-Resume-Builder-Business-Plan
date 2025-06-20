import streamlit as st
from agents.brainstorm import generate_prompts, get_project_metadata, memory as brainstorm_memory
from agents.idea_capture import process_ideas
from agents.challenge_detector import detect_challenges
from agents.opportunity_scorer import score_opportunities
from agents.problem_refiner import refine_problem_statement
from config import GOOGLE_API_KEY, GEMINI_MODEL, VECTORSTORE_PATH

# Streamlit page configuration
st.set_page_config(page_title="Agentic AI Workflow for Student Projects", layout="wide")

# Title and description
st.title("Agentic AI Workflow for Student Projects")
st.markdown("""
This application uses AI-driven agents to generate, organize, analyze, score, and refine project ideas.
1. Generate brainstorming prompts.
2. Cluster ideas and suggest diagrams.
3. Identify user-centered challenges.
4. Score and rank opportunities.
5. Refine a final problem statement and knowledge plan.
""")

# Initialize session state
if "metadata" not in st.session_state:
    st.session_state.metadata = None
if "prompts" not in st.session_state:
    st.session_state.prompts = []
if "clustered_ideas" not in st.session_state:
    st.session_state.clustered_ideas = {}
if "diagram_suggestions" not in st.session_state:
    st.session_state.diagram_suggestions = []
if "challenges" not in st.session_state:
    st.session_state.challenges = []
if "filtered_ideas" not in st.session_state:
    st.session_state.filtered_ideas = []
if "ranked_opportunities" not in st.session_state:
    st.session_state.ranked_opportunities = []
if "problem_statement" not in st.session_state:
    st.session_state.problem_statement = ""
if "knowledge_plan" not in st.session_state:
    st.session_state.knowledge_plan = {}

# Step 1: Brainstorm Prompts
st.header("Step 1: Generate Brainstorming Prompts")
with st.form(key="metadata_form"):
    title = st.text_input("Project Title", placeholder="e.g., Agentic AI Framework")
    theme = st.text_input("Theme", placeholder="e.g., agentic AI")
    department = st.text_input("Department", placeholder="e.g., Computer Science")
    submit_button = st.form_submit_button(label="Generate Prompts")

    if submit_button:
        if title and theme and department:
            st.session_state.metadata = get_project_metadata(title, theme, department)
            try:
                st.session_state.prompts = generate_prompts(st.session_state.metadata)
                st.success("Prompts generated successfully!")
            except Exception as e:
                st.error(f"Error generating prompts: {e}. Check logs and GOOGLE_API_KEY.")
        else:
            st.error("Please fill in all metadata fields.")

if st.session_state.prompts:
    st.subheader("Generated Brainstorming Prompts")
    for i, prompt in enumerate(st.session_state.prompts, 1):
        st.markdown(f"**{i}.** {prompt}")

# Step 2: Idea Capturing
st.header("Step 2: Capture and Organize Ideas")
with st.form(key="ideas_form"):
    default_ideas = "\n".join([p.split("(Source:")[0].replace("Prompt:", "").strip() for p in st.session_state.prompts]) if st.session_state.prompts else ""
    ideas_input = st.text_area("Enter Your Ideas (one per line)", default_ideas, placeholder="e.g., Design a multi-agent system...")
    submit_ideas_button = st.form_submit_button(label="Process Ideas")

    if submit_ideas_button:
        if ideas_input and st.session_state.metadata:
            ideas = [idea.strip() for idea in ideas_input.split("\n") if idea.strip()]
            try:
                result = process_ideas(ideas, st.session_state.metadata.get("theme", "general"))
                if "error" in result:
                    st.error(f"Error processing ideas: {result['error']}. Check GOOGLE_API_KEY and GEMINI_MODEL in config.py.")
                    st.session_state.clustered_ideas = {}
                    st.session_state.diagram_suggestions = []
                else:
                    st.session_state.clustered_ideas = result["clustered_ideas"]
                    st.session_state.diagram_suggestions = result["diagram_suggestions"]
                    if not any(st.session_state.clustered_ideas.values()):
                        st.warning("No valid clusters generated. Try more specific ideas or check LLM configuration.")
                    else:
                        st.success("Ideas processed successfully!")
            except Exception as e:
                st.error(f"Error processing ideas: {e}. Check logs for details.")
                st.session_state.clustered_ideas = {}
                st.session_state.diagram_suggestions = []
        else:
            st.error("Please enter ideas and complete Step 1.")

if st.session_state.clustered_ideas:
    st.subheader("Clustered Ideas")
    for theme, ideas in st.session_state.clustered_ideas.items():
        st.markdown(f"**Theme: {theme}**")
        for item in ideas:
            st.markdown(f"- **Idea**: {item['idea']}, **Tags**: {', '.join(item['tags'])}")

if st.session_state.diagram_suggestions:
    st.subheader("Diagram Suggestions")
    for suggestion in st.session_state.diagram_suggestions:
        st.markdown(f"- {suggestion}")

# Step 3: Challenge Detection
st.header("Step 3: Detect User-Centered Challenges")
with st.form(key="challenges_form"):
    submit_challenges_button = st.form_submit_button(label="Analyze Challenges")

    if submit_challenges_button:
        if st.session_state.clustered_ideas and any(st.session_state.clustered_ideas.values()) and st.session_state.metadata:
            try:
                result = detect_challenges(st.session_state.clustered_ideas, st.session_state.metadata.get("theme", "general"))
                if "error" in result:
                    st.error(f"Error analyzing challenges: {result['error']}. Check logs or GOOGLE_API_KEY.")
                    st.session_state.challenges = []
                    st.session_state.filtered_ideas = []
                else:
                    st.session_state.challenges = result["challenges"]
                    st.session_state.filtered_ideas = result["filtered_ideas"]
                    if not result["challenges"] and not result["filtered_ideas"]:
                        st.warning("No challenges or filtered ideas generated. Try more specific ideas or check LLM configuration.")
                    else:
                        st.success("Challenges analyzed successfully!")
            except Exception as e:
                st.error(f"Error analyzing challenges: {e}. Check logs or GOOGLE_API_KEY.")
                st.session_state.challenges = []
                st.session_state.filtered_ideas = []
        else:
            st.error("Please complete Step 2 with valid clustered ideas first.")

if st.session_state.challenges:
    st.subheader("Detected Challenges")
    for challenge in st.session_state.challenges:
        st.markdown(f"- **Idea**: {challenge['idea']}")
        st.markdown(f"  **Pain Point**: {challenge.get('pain_point', 'N/A')}")
        st.markdown(f"  **Affected Users/Stakeholders**: {challenge.get('users_stakeholders', 'N/A')}")
        st.markdown(f"  **Importance**: {challenge.get('importance', 'N/A')}")

if st.session_state.filtered_ideas:
    st.subheader("Filtered Ideas")
    for filtered in st.session_state.filtered_ideas:
        st.markdown(f"- **Idea**: {filtered['idea']}, **Reason**: {filtered['reason']}")

# Step 4: Opportunity Scoring and Ranking
st.header("Step 4: Score and Rank Opportunities")
with st.form(key="opportunities_form"):
    survey_results = st.text_area("Enter Survey Results (optional, one per line, format: Idea: [idea], User Rating: [score]/100)", placeholder="e.g., Idea: Optimize a process in agentic AI, User Rating: 85/100")
    submit_opportunities_button = st.form_submit_button(label="Score Opportunities")

    if submit_opportunities_button:
        if st.session_state.challenges and st.session_state.metadata:
            try:
                result = score_opportunities(st.session_state.challenges, st.session_state.metadata.get("theme", "general"), survey_results)
                if "error" in result:
                    st.error(f"Error scoring opportunities: {result['error']}. Check logs, GOOGLE_API_KEY, or langchain version.")
                    st.session_state.ranked_opportunities = []
                else:
                    st.session_state.ranked_opportunities = result["ranked_opportunities"]
                    if not result["ranked_opportunities"]:
                        st.warning("No opportunities ranked. Try more specific challenges or check LLM configuration.")
                    else:
                        st.success("Opportunities scored and ranked successfully!")
            except Exception as e:
                st.error(f"Error scoring opportunities: {e}. Check logs or GOOGLE_API_KEY.")
                st.session_state.ranked_opportunities = []
        else:
            st.error("Please complete Step 3 with valid challenges first.")

if st.session_state.ranked_opportunities:
    st.subheader("Ranked Opportunities")
    for opp in st.session_state.ranked_opportunities:
        st.markdown(f"- **{opp['rank']}: {opp['idea']}**")
        st.markdown(f"  **Feasibility**: {opp.get('feasibility', 'N/A')}/100 (Reason: {opp.get('feasibility_reason', 'N/A')})")
        st.markdown(f"  **Impact**: {opp.get('impact', 'N/A')}/100 (Reason: {opp.get('impact_reason', 'N/A')})")
        st.markdown(f"  **Empathy**: {opp.get('empathy', 'N/A')}/100 (Reason: {opp.get('empathy_reason', 'N/A')})")
        st.markdown(f"  **Total Score**: {opp.get('total_score', 'N/A')}/100")

# Step 5: Problem Statement Refinement
st.header("Step 5: Refine Problem Statement and Knowledge Plan")
with st.form(key="problem_refiner_form"):
    submit_refiner_button = st.form_submit_button(label="Refine Problem Statement")

    if submit_refiner_button:
        if st.session_state.ranked_opportunities and st.session_state.metadata:
            top_opportunity = st.session_state.ranked_opportunities[0]  # Take Rank 1
            try:
                result = refine_problem_statement(top_opportunity, st.session_state.metadata.get("theme", "general"))
                if "error" in result:
                    st.error(f"Error refining problem statement: {result['error']}. Check logs, GOOGLE_API_KEY, or langchain version.")
                    st.session_state.problem_statement = ""
                    st.session_state.knowledge_plan = {}
                else:
                    st.session_state.problem_statement = result["problem_statement"]
                    st.session_state.knowledge_plan = result["knowledge_plan"]
                    if not result["problem_statement"]:
                        st.warning("No problem statement generated. Check LLM configuration or top opportunity details.")
                    else:
                        st.success("Problem statement and knowledge plan generated successfully!")
            except Exception as e:
                st.error(f"Error refining problem statement: {e}. Check logs or GOOGLE_API_KEY.")
                st.session_state.problem_statement = ""
                st.session_state.knowledge_plan = {}
        else:
            st.error("Please complete Step 4 with valid ranked opportunities first.")

if st.session_state.problem_statement:
    st.subheader("Refined Problem Statement")
    st.markdown(f"{st.session_state.problem_statement}")

if st.session_state.knowledge_plan:
    st.subheader("Knowledge Plan")
    st.markdown("**What we know:**")
    for point in st.session_state.knowledge_plan.get("what_we_know", []):
        st.markdown(f"- {point}")
    st.markdown("**What we need to learn:**")
    for point in st.session_state.knowledge_plan.get("what_we_need_to_learn", []):
        st.markdown(f"- {point}")

# Debug memory
if st.button("Show Conversation Memory"):
    st.subheader("Conversation Memory")
    brainstorm_memory_content = brainstorm_memory.load_memory_variables({})
    from agents.idea_capture import memory as idea_memory
    idea_memory_content = idea_memory.load_memory_variables({})
    from agents.challenge_detector import memory as challenge_memory
    challenge_memory_content = challenge_memory.load_memory_variables({})
    from agents.opportunity_scorer import memory as opportunity_memory
    opportunity_memory_content = opportunity_memory.load_memory_variables({})
    from agents.problem_refiner import memory as refiner_memory
    refiner_memory_content = refiner_memory.load_memory_variables({})
    st.write("**Brainstorm Agent Memory**:", brainstorm_memory_content)
    st.write("**Idea Capture Agent Memory**:", idea_memory_content)
    st.write("**Challenge Detector Agent Memory**:", challenge_memory_content)
    st.write("**Opportunity Scorer Agent Memory**:", opportunity_memory_content)
    st.write("**Problem Refiner Agent Memory**:", refiner_memory_content)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, LangChain, and Google's Gemini API.")
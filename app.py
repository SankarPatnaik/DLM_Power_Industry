import os
from datetime import datetime, timedelta

import importlib
import importlib.util

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Agentic AI for Power Grid", layout="wide")

st.title("⚡ Agentic AI Control Room for the Power Grid")
st.subheader("CrewAI-powered multi-agent decision support for utilities")
st.markdown(
    "This app uses specialized AI agents (Operations, Reliability, and Maintenance) "
    "to analyze grid events and provide coordinated recommendations."
)

POWER_GRID_TOPICS = {
    "Grid Reliability": [
        "Which asset presents the highest N-1 reliability risk right now?",
        "What contingency should operators prioritize based on the current snapshot?",
    ],
    "Outage & Restoration": [
        "Why did feeder F12 trip and what should restoration sequencing look like?",
        "How can we reduce customer outage duration if transformer T3 worsens?",
    ],
    "Renewable Integration": [
        "How does the wind generation drop affect balancing and reserve needs?",
        "What immediate actions stabilize the grid if renewable output falls further?",
    ],
    "Asset Health & Maintenance": [
        "Which equipment needs immediate inspection within 6 hours?",
        "What maintenance plan lowers repeat trips over the next 48 hours?",
    ],
    "Load Management": [
        "Is there overload risk growth in the next few hours and how should we respond?",
        "What demand response actions could reduce stress on critical assets?",
    ],
}

# Sample operational telemetry
sample_data = {
    "Asset": [
        "Transformer T3",
        "Feeder F12",
        "Substation A",
        "Wind Farm 2",
        "Solar Plant 5",
    ],
    "Status": ["Overload", "Tripped", "Stable", "Generation Drop", "Stable"],
    "Load_%": [96, 0, 64, 42, 51],
    "Last_Event_Time": [
        datetime.now() - timedelta(hours=2),
        datetime.now() - timedelta(hours=5),
        datetime.now() - timedelta(hours=1),
        datetime.now() - timedelta(hours=3),
        datetime.now() - timedelta(hours=6),
    ],
}

df = pd.DataFrame(sample_data)

with st.expander("Live Grid Snapshot", expanded=True):
    st.dataframe(df, width="stretch")


def format_context(dataframe: pd.DataFrame) -> str:
    """Format telemetry table into a compact context string for agents."""
    rows = []
    for _, row in dataframe.iterrows():
        rows.append(
            f"Asset={row['Asset']}, Status={row['Status']}, Load={row['Load_%']}%, "
            f"LastEvent={row['Last_Event_Time']}"
        )
    return "\n".join(rows)


def build_crew(user_question: str, grid_context: str, topic_focus: str, urgency: str):
    """Build an agentic CrewAI workflow for power-grid analysis."""
    if importlib.util.find_spec("crewai") is None:
        raise RuntimeError("CrewAI is not installed. Run `pip install -r requirements.txt`.")

    crewai = importlib.import_module("crewai")
    Agent = crewai.Agent
    Crew = crewai.Crew
    Process = crewai.Process
    Task = crewai.Task

    ops_agent = Agent(
        role="Grid Operations Analyst",
        goal="Detect what happened in the grid and identify operational root causes quickly.",
        backstory=(
            "You are a utility control-room expert. You interpret alarms, feeder events, "
            "and load trends to explain incidents."
        ),
        allow_delegation=False,
        verbose=False,
    )

    reliability_agent = Agent(
        role="Grid Reliability Engineer",
        goal="Assess system risk and reliability impact from current events.",
        backstory=(
            "You specialize in N-1 reliability, contingency analysis, and outage prevention "
            "for transmission and distribution networks."
        ),
        allow_delegation=False,
        verbose=False,
    )

    maintenance_agent = Agent(
        role="Asset Maintenance Planner",
        goal="Recommend practical maintenance and mitigation actions with urgency levels.",
        backstory=(
            "You optimize utility maintenance plans using asset condition, trip history, "
            "and operational stress indicators."
        ),
        allow_delegation=False,
        verbose=False,
    )

    operations_task = Task(
        description=(
            "Analyze the user question and grid telemetry context. "
            "Identify the most likely operational cause and supporting evidence.\n\n"
            f"Topic Focus: {topic_focus}\n"
            f"Urgency: {urgency}\n\n"
            f"User Question: {user_question}\n\n"
            f"Grid Context:\n{grid_context}"
        ),
        expected_output="Concise operational diagnosis with evidence from telemetry.",
        agent=ops_agent,
    )

    reliability_task = Task(
        description=(
            "Using the operational diagnosis, evaluate reliability impact. "
            "Rate risk as Low/Medium/High and explain outage implications. "
            "Include how urgency should influence operator decisions."
        ),
        expected_output="Risk rating and reliability impact summary.",
        agent=reliability_agent,
        context=[operations_task],
    )

    maintenance_task = Task(
        description=(
            "Using previous analyses, produce an action plan with immediate (0-6h), "
            "short-term (24-48h), and follow-up actions. "
            "Add one clarifying follow-up question to keep the operator interaction going."
        ),
        expected_output=(
            "Prioritized maintenance and mitigation action plan with one follow-up question."
        ),
        agent=maintenance_agent,
        context=[operations_task, reliability_task],
    )

    return Crew(
        agents=[ops_agent, reliability_agent, maintenance_agent],
        tasks=[operations_task, reliability_task, maintenance_task],
        process=Process.sequential,
        verbose=False,
    )


with st.sidebar:
    st.header("Configuration")
    st.markdown("Set your OpenAI API key as an environment variable before running:")
    st.code("export OPENAI_API_KEY='your_key_here'", language="bash")
    st.caption("CrewAI will use your configured LLM provider credentials.")

    st.markdown("---")
    topic_focus = st.selectbox("Topic focus", list(POWER_GRID_TOPICS.keys()), index=0)
    urgency = st.radio("Operational urgency", ["Routine", "Elevated", "Critical"], index=1)

st.markdown("### Ask an Interactive Grid Question")
st.caption("Choose a starter question below or write your own.")

selected_starter = st.selectbox(
    "Suggested power-grid questions",
    ["Custom question..."] + POWER_GRID_TOPICS[topic_focus],
)

default_question = "" if selected_starter == "Custom question..." else selected_starter
question = st.text_area(
    "Your question",
    value=default_question,
    placeholder="Example: What actions should we take if transformer T3 load exceeds 98%?",
    height=110,
)

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if st.button("Run Multi-Agent Analysis", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Please add your key and restart Streamlit.")
    else:
        context_text = format_context(df)
        crew = build_crew(question, context_text, topic_focus, urgency)

        with st.spinner("Agents are collaborating on your request..."):
            try:
                result = crew.kickoff()
                answer = str(result)
                st.success("Analysis complete")
                st.markdown("### CrewAI Final Response")
                st.write(answer)

                st.session_state.analysis_history.insert(
                    0,
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "topic": topic_focus,
                        "urgency": urgency,
                        "question": question,
                        "answer": answer,
                    },
                )
            except Exception as exc:
                st.error(
                    "CrewAI execution failed. Confirm dependencies and API key configuration."
                )
                st.exception(exc)

if st.session_state.analysis_history:
    st.markdown("### Previous Q&A")
    for item in st.session_state.analysis_history[:5]:
        with st.expander(f"{item['timestamp']} • {item['topic']} • {item['urgency']}"):
            st.markdown(f"**Question:** {item['question']}")
            st.markdown("**Answer:**")
            st.write(item["answer"])

st.markdown("---")
st.caption("Built with Streamlit + CrewAI for agentic utility operations intelligence")

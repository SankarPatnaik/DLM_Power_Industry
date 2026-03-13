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


def build_crew(user_question: str, grid_context: str):
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
            f"User Question: {user_question}\n\n"
            f"Grid Context:\n{grid_context}"
        ),
        expected_output="Concise operational diagnosis with evidence from telemetry.",
        agent=ops_agent,
    )

    reliability_task = Task(
        description=(
            "Using the operational diagnosis, evaluate reliability impact. "
            "Rate risk as Low/Medium/High and explain outage implications."
        ),
        expected_output="Risk rating and reliability impact summary.",
        agent=reliability_agent,
        context=[operations_task],
    )

    maintenance_task = Task(
        description=(
            "Using previous analyses, produce an action plan with immediate (0-6h), "
            "short-term (24-48h), and follow-up actions."
        ),
        expected_output="Prioritized maintenance and mitigation action plan.",
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

question = st.text_input("Ask a grid question:", placeholder="Why did feeder F12 trip?")

if st.button("Run Multi-Agent Analysis", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Please add your key and restart Streamlit.")
    else:
        context_text = format_context(df)
        crew = build_crew(question, context_text)

        with st.spinner("Agents are collaborating on your request..."):
            try:
                result = crew.kickoff()
                st.success("Analysis complete")
                st.markdown("### CrewAI Final Response")
                st.write(str(result))
            except Exception as exc:
                st.error(
                    "CrewAI execution failed. Confirm dependencies and API key configuration."
                )
                st.exception(exc)

st.markdown("---")
st.caption("Built with Streamlit + CrewAI for agentic utility operations intelligence")

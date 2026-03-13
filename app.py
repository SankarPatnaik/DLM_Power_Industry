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

HISTORICAL_FAILURE_DATA = [
    {
        "Date": "2026-01-14 03:12",
        "Asset": "Feeder F12",
        "Failure_Type": "Protection Trip",
        "Root_Cause": "Insulation degradation caused phase-to-ground fault during fog event",
        "Contributing_Factors": "Moisture ingress, delayed cable joint replacement, high morning load ramp",
        "Customers_Affected": 12840,
        "Duration_Min": 96,
        "Corrective_Action": "Replaced damaged cable section, retuned relay pickup, accelerated joint replacement backlog",
    },
    {
        "Date": "2026-02-03 17:48",
        "Asset": "Transformer T3",
        "Failure_Type": "Thermal Overload",
        "Root_Cause": "Cooling fan bank B failed while evening peak exceeded forecast by 11%",
        "Contributing_Factors": "Deferred fan motor maintenance, low reactive support, sustained 97-101% loading",
        "Customers_Affected": 9230,
        "Duration_Min": 64,
        "Corrective_Action": "Repaired fan bank, temporary load transfer to T4, added thermal alarm threshold at 92%",
    },
    {
        "Date": "2026-02-19 11:05",
        "Asset": "Wind Farm 2",
        "Failure_Type": "Generation Curtailment",
        "Root_Cause": "Converter control board fault triggered emergency shutdown",
        "Contributing_Factors": "Harmonic distortion from nearby industrial feeder, missed firmware patch",
        "Customers_Affected": 0,
        "Duration_Min": 142,
        "Corrective_Action": "Swapped control board, deployed firmware update, added power quality monitoring",
    },
    {
        "Date": "2026-03-01 22:33",
        "Asset": "Substation A",
        "Failure_Type": "Bus Undervoltage Event",
        "Root_Cause": "Capacitor bank breaker failed to close during contingency switching",
        "Contributing_Factors": "Aged breaker mechanism, incomplete night-shift checklist execution",
        "Customers_Affected": 6740,
        "Duration_Min": 38,
        "Corrective_Action": "Breaker overhaul, mandatory switching checklist sign-off, added remote close test",
    },
]

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
historical_df = pd.DataFrame(HISTORICAL_FAILURE_DATA)
historical_df["Date"] = pd.to_datetime(historical_df["Date"])

with st.expander("Live Grid Snapshot", expanded=True):
    st.dataframe(df, width="stretch")

with st.expander("Failure History (last 90 days)"):
    st.caption(
        "Synthetic incident data to let agents perform root-cause lookbacks and explain past failures."
    )
    st.dataframe(
        historical_df.sort_values("Date", ascending=False).reset_index(drop=True),
        width="stretch",
    )


def format_context(dataframe: pd.DataFrame) -> str:
    """Format telemetry table into a compact context string for agents."""
    rows = []
    for _, row in dataframe.iterrows():
        rows.append(
            f"Asset={row['Asset']}, Status={row['Status']}, Load={row['Load_%']}%, "
            f"LastEvent={row['Last_Event_Time']}"
        )
    return "\n".join(rows)


def build_failure_history_context(user_question: str, data: pd.DataFrame, max_rows: int = 3) -> str:
    """Return the most relevant historical incidents for the current question."""
    lowered_question = user_question.lower()
    scored_rows = []
    for _, row in data.iterrows():
        score = 0
        text_blob = " ".join(str(value).lower() for value in row.values)

        if str(row["Asset"]).lower() in lowered_question:
            score += 5
        if str(row["Failure_Type"]).lower() in lowered_question:
            score += 3
        if "past" in lowered_question or "history" in lowered_question or "previous" in lowered_question:
            score += 2
        if any(keyword in lowered_question for keyword in ["why", "cause", "failure", "trip"]):
            score += 2
        if any(token in text_blob for token in lowered_question.split()):
            score += 1

        scored_rows.append((score, row))

    ranked = sorted(scored_rows, key=lambda item: (item[0], item[1]["Date"]), reverse=True)
    top_rows = [item[1] for item in ranked[:max_rows]]

    history_lines = []
    for row in top_rows:
        history_lines.append(
            " | ".join(
                [
                    f"Date={row['Date']:%Y-%m-%d %H:%M}",
                    f"Asset={row['Asset']}",
                    f"FailureType={row['Failure_Type']}",
                    f"RootCause={row['Root_Cause']}",
                    f"ContributingFactors={row['Contributing_Factors']}",
                    f"DurationMin={row['Duration_Min']}",
                    f"CorrectiveAction={row['Corrective_Action']}",
                ]
            )
        )

    return "\n".join(history_lines)


def normalize_model_name(provider: str, model_name: str) -> str:
    """Normalize model names to provider-prefixed LiteLLM format."""
    cleaned = model_name.strip()
    if not cleaned:
        return cleaned

    if "/" in cleaned:
        return cleaned

    provider_prefix = {
        "OpenAI": "openai",
        "Groq": "groq",
        "Vertex AI": "vertex_ai",
    }.get(provider)

    return f"{provider_prefix}/{cleaned}" if provider_prefix else cleaned


def build_crew(
    user_question: str,
    grid_context: str,
    failure_history_context: str,
    topic_focus: str,
    urgency: str,
    provider: str,
    llm_model: str,
):
    """Build an agentic CrewAI workflow for power-grid analysis."""
    if importlib.util.find_spec("crewai") is None:
        raise RuntimeError("CrewAI is not installed. Run `pip install -r requirements.txt`.")

    crewai = importlib.import_module("crewai")
    Agent = crewai.Agent
    Crew = crewai.Crew
    LLM = getattr(crewai, "LLM", None)
    Process = crewai.Process
    Task = crewai.Task

    normalized_model = normalize_model_name(provider, llm_model)
    llm = LLM(model=normalized_model) if LLM else normalized_model

    ops_agent = Agent(
        role="Grid Operations Analyst",
        goal="Detect what happened in the grid and identify operational root causes quickly.",
        backstory=(
            "You are a utility control-room expert. You interpret alarms, feeder events, "
            "and load trends to explain incidents."
        ),
        allow_delegation=False,
        verbose=False,
        llm=llm,
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
        llm=llm,
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
        llm=llm,
    )

    operations_task = Task(
        description=(
            "Analyze the user question and grid telemetry context. "
            "Identify the most likely operational cause and supporting evidence. "
            "If question asks about previous incidents, explicitly reference failure history records.\n\n"
            f"Topic Focus: {topic_focus}\n"
            f"Urgency: {urgency}\n\n"
            f"User Question: {user_question}\n\n"
            f"Grid Context:\n{grid_context}\n\n"
            f"Relevant Failure History:\n{failure_history_context}"
        ),
        expected_output=(
            "Concise operational diagnosis with evidence from telemetry and, when relevant,"
            " a comparison to past failure records."
        ),
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

    provider = st.selectbox(
        "LLM provider",
        ["OpenAI", "Groq", "Vertex AI"],
        index=0,
        help="Choose the provider you have credentials for.",
    )

    if provider == "OpenAI":
        llm_model = st.text_input("Model", value="openai/gpt-4o-mini")
        st.markdown("Set your OpenAI API key:")
        st.code("export OPENAI_API_KEY='your_key_here'", language="bash")
        provider_ready = bool(os.getenv("OPENAI_API_KEY"))
        missing_message = "OPENAI_API_KEY is not set. Please add your key and restart Streamlit."
    elif provider == "Groq":
        llm_model = st.text_input("Model", value="groq/llama-3.1-70b-versatile")
        st.markdown("Set your Groq key:")
        st.code("export GROQ_API_KEY='your_key_here'", language="bash")
        provider_ready = bool(os.getenv("GROQ_API_KEY"))
        missing_message = "GROQ_API_KEY is not set. Please add your key and restart Streamlit."
    else:
        llm_model = st.text_input("Model", value="vertex_ai/gemini-1.5-pro")
        st.markdown("Set your Vertex AI credentials:")
        st.code(
            "\n".join(
                [
                    "export GOOGLE_APPLICATION_CREDENTIALS='/path/to/service_account.json'",
                    "export VERTEXAI_PROJECT='your_gcp_project_id'",
                    "export VERTEXAI_LOCATION='us-central1'",
                ]
            ),
            language="bash",
        )
        provider_ready = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")) and bool(
            os.getenv("VERTEXAI_PROJECT")
        )
        missing_message = (
            "Vertex AI credentials are not fully set. "
            "Expected GOOGLE_APPLICATION_CREDENTIALS and VERTEXAI_PROJECT "
            "(optional: VERTEXAI_LOCATION)."
        )

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
    elif not provider_ready:
        st.error(missing_message)
    else:
        context_text = format_context(df)
        failure_history_context = build_failure_history_context(question, historical_df)
        crew = build_crew(
            question,
            context_text,
            failure_history_context,
            topic_focus,
            urgency,
            provider,
            llm_model,
        )

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
                    "CrewAI execution failed. Confirm dependencies and API key/model configuration."
                )
                if "Fallback to LiteLLM is not available" in str(exc):
                    st.info(
                        "Install LiteLLM and use a provider-prefixed model name like "
                        "`openai/gpt-4o-mini` or `groq/llama-3.1-70b-versatile`."
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

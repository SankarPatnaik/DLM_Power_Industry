# Agentic AI Power Grid Control Room (CrewAI + Streamlit)

This project converts a basic power-grid Q&A demo into an **agentic multi-agent system** using **CrewAI**.

## What it does

The Streamlit app now runs a team of specialized AI agents:

1. **Grid Operations Analyst** – diagnoses what happened.
2. **Grid Reliability Engineer** – evaluates reliability risk.
3. **Asset Maintenance Planner** – proposes prioritized actions.

The crew processes your question and the grid telemetry table sequentially, then returns a combined recommendation.

## Prerequisites

- Python 3.10+
- One provider credential set: OpenAI API key, Groq API key, or Vertex AI service account

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your_openai_api_key"
# OR
export GROQ_API_KEY="your_groq_api_key"
# OR
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service_account.json"
export VERTEXAI_PROJECT="your_gcp_project_id"
export VERTEXAI_LOCATION="us-central1"
```



## Provider support

The app sidebar now supports switching between:

- **OpenAI** (default model: `openai/gpt-4o-mini`)
- **Groq** (default model: `groq/llama-3.1-70b-versatile`)
- **Vertex AI** (default model: `vertex_ai/gemini-1.5-pro`)

Pick a provider, set the matching environment variables, and run the analysis.

## Run

```bash
streamlit run app.py
```

## Example questions

- Why did feeder F12 trip?
- Is there overload risk in transformer T3?
- What should maintenance prioritize in the next 48 hours?
- What is the reliability impact of current events?

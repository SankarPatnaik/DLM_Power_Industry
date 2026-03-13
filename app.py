
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="AI Brain for the Power Grid", layout="wide")

st.title("⚡ AI Brain for the Power Grid")
st.subheader("Domain Language Model Demo for Power Utilities")

st.markdown("Ask the AI about grid events, overload risk, or maintenance insights.")

data = {
    "Asset": ["Transformer T3", "Feeder F12", "Substation A", "Wind Farm 2", "Solar Plant 5"],
    "Status": ["Overload", "Tripped", "Stable", "Generation Drop", "Stable"],
    "Last_Event_Time": [
        datetime.now() - timedelta(hours=2),
        datetime.now() - timedelta(hours=5),
        datetime.now() - timedelta(hours=1),
        datetime.now() - timedelta(hours=3),
        datetime.now() - timedelta(hours=6),
    ],
}

df = pd.DataFrame(data)

st.dataframe(df)

question = st.text_input("Ask a question about the grid:")

responses = {
    "trip": "Feeder F12 tripped due to overcurrent caused by sudden industrial demand spike.",
    "overload": "Transformer T3 is currently running at 96% capacity. Recommend load transfer to Substation B.",
    "maintenance": "Transformer T3 cooling system shows degradation. Maintenance recommended within 48 hours.",
    "grid": "Overall grid stability is 94%. No critical outages expected tonight.",
}

def generate_answer(q):
    q = q.lower()
    for key in responses:
        if key in q:
            return responses[key]
    return "AI analysis: Grid is stable. No major anomalies detected."

if question:
    st.success(generate_answer(question))

st.markdown("---")
st.caption("Demo concept: Domain Language Model for Power Grid Operations")

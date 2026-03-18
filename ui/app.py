from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("INSURANCE_AI_API", "http://localhost:8000/ask")

st.set_page_config(page_title="Insurance AI Assistant", layout="wide")
st.title("🤖 Insurance AI Assistant")
st.write("Ask questions about policies, claims, coverage, exclusions, and claim workflows.")

query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            try:
                response = requests.get(API_URL, params={"query": query}, timeout=120)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                st.error(
                    "Request failed. Make sure the backend is running and the API URL is correct."
                )
                st.write(f"Error: {e}")
            else:
                data = response.json()
                st.success(data["answer"])
                if data.get("sources"):
                    with st.expander("Sources"):
                        for source in data["sources"]:
                            st.markdown(f"**{source['source']}**")
                            st.caption(source["excerpt"])
    else:
        st.warning("Please enter a question before submitting.")

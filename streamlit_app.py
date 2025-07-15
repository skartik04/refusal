# streamlit_app.py

import streamlit as st
from llm_hooks import run_with_mode

st.title("🧠 Intervention-Based LLM Output Viewer")

mode = st.selectbox("Choose intervention mode:", ["refuse", "bypass"])
prompt = st.text_area("Enter your prompt:")

if st.button("Generate"):
    with st.spinner("Generating completions..."):
        baseline, intervention = run_with_mode(prompt, mode)
        st.markdown("### Baseline Response")
        st.success(baseline)
        st.markdown(f"### {'Refusal' if mode=='refuse' else 'Bypass'} Intervention Response")
        st.warning(intervention)

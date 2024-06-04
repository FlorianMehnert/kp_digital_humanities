from langchain_community.llms import Ollama

import streamlit as st

llm = Ollama(model="llama3")

st.title("Chatbot using Llama3")
prompt = st.text_area("Enter your prompt:")

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response..."):
            st.write(llm.invoke(prompt, stop=['<|eot_id|>']))


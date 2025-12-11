import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="LangChain Chatbot", layout="centered")
st.title("💬 AI Agent")

API_CHAT_URL = os.getenv("API_CHAT_URL", "http://localhost:8000/chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# input
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    try:
        with st.spinner("Thinking..."):
            res = requests.post(API_CHAT_URL, json={"message": prompt}, timeout=60)
            res.raise_for_status()
            data = res.json()
            reply = data.get("result") or data.get("answer") or str(data)

            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.write(reply)
    except Exception as e:
        st.error(f"❌ Error: {e}")

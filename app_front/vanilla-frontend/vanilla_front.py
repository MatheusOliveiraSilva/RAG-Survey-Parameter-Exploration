import streamlit as st
from app_front.utils.msg_utils import get_response
from agent.graphs.simple_rag import graph
from langchain_core.messages import HumanMessage
import time

st.title("Simple RAG Chatbot")

memory_config = {"configurable": {"thread_id": "1"}}

# Streamed response emulator
def response_generator(user_input, config):
    messages = [HumanMessage(
        content=user_input)
    ]

    for msg, metadata in graph.stream(
            {"messages": messages},
            stream_mode="messages", config=config
    ):
        if msg.content:
            yield msg.content
            time.sleep(0.05)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt, memory_config))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

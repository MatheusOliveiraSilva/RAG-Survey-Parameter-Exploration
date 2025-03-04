import streamlit as st
from app_front.utils.msg_utils import get_response
from agent.graphs.simple_rag import graph
from langchain_core.messages import HumanMessage

st.title("Echo Bot")

memory_config = {"configurable": {"thread_id": "1"}}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    messages = [HumanMessage(content=prompt)]
    result = graph.invoke({"messages": messages}, memory_config)
    response = get_response(result, 'anthropic')

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

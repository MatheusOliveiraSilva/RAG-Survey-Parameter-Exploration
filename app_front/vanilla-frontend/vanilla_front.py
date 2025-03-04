import streamlit as st
import time
from langchain_core.messages import HumanMessage, AIMessageChunk
from agent.graphs.simple_rag import graph

st.title("Simple RAG Chatbot")

# Configuração de memória e thread
memory_config = {"configurable": {"thread_id": "1"}}

# Inicializa os históricos, se ainda não existirem
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thoughts" not in st.session_state:
    st.session_state.thoughts = ""

# Exibe o histórico de mensagens no chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Se já houver pensamentos salvos, exibe-os em um expander (colapsado)
if st.session_state.thoughts:
    with st.expander("🤖 Pensamentos do modelo", expanded=False):
        st.markdown(st.session_state.thoughts)

# Recebe a entrada do usuário
if prompt := st.chat_input("O que deseja perguntar?"):
    # Adiciona a mensagem do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Inicializa variáveis para streaming
    final_response = ""
    streaming_thoughts = ""

    # Cria placeholders para atualizar o conteúdo em tempo real
    final_placeholder = st.empty()      # para a resposta final
    thinking_placeholder = st.empty()     # para os pensamentos durante o streaming

    # Streaming: atualiza os placeholders conforme os chunks chegam
    for response in graph.stream(
            {"messages": [HumanMessage(content=prompt)]},
            stream_mode="messages",
            config=memory_config
    ):
        # Como o streaming agora vem em tuplas contendo AIMessageChunk,
        # iteramos sobre cada item da tupla:
        if isinstance(response, tuple):
            for item in response:
                if isinstance(item, AIMessageChunk) and item.content:
                    # O item.content é uma lista de dicionários
                    chunk = item.content[0]
                    if "type" in chunk:
                        if chunk["type"] == "thinking" and "thinking" in chunk:
                            streaming_thoughts += chunk["thinking"]
                            # Atualiza a área de pensamento durante o streaming
                            thinking_placeholder.markdown(
                                f"**Pensamentos em streaming:**\n\n{streaming_thoughts}"
                            )
                        elif chunk["type"] == "text" and "text" in chunk:
                            final_response += chunk["text"]
                            # Atualiza a resposta final em streaming no chat
                            final_placeholder.markdown(final_response)
        time.sleep(0.05)

    # Quando o streaming terminar, remove o placeholder de pensamentos
    thinking_placeholder.empty()
    # Armazena os pensamentos na sessão e os exibe em um expander (colapsado)
    if streaming_thoughts.strip():
        st.session_state.thoughts += streaming_thoughts
        with st.expander("🤖 Pensamentos do modelo", expanded=False):
            st.markdown(st.session_state.thoughts)

    # Salva a resposta final no histórico e exibe no chat
    st.session_state.messages.append({"role": "assistant", "content": final_response})

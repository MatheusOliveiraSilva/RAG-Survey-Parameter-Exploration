import time
from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from agent.nodes.nodes import SimpleRAG
from agent.states.simple_rag_state import SimpleRagState

# Inicializa o agente
SimpleRAG = SimpleRAG(llm_provider="anthropic", llm_model_name="claude-3-7-sonnet-latest")

memory = MemorySaver()

# Constrói o grafo
builder = StateGraph(SimpleRagState)
builder.add_node("retrieval", SimpleRAG.retrieval)
builder.add_node("assistant", SimpleRAG.assistant)
builder.add_edge(START, "retrieval")
builder.add_edge("retrieval", "assistant")
graph = builder.compile(checkpointer=memory)

def get_response(msg):
    messages = [HumanMessage(content=msg)]
    thoughts = ""
    final_response = ""

    for response in graph.stream({"messages": messages}, stream_mode="messages", config={"configurable": {"thread_id": "1"}}):
        if isinstance(response, tuple):
            for item in response:
                if isinstance(item, AIMessageChunk) and item.content and "type" in item.content[0]:
                    if item.content[0]["type"] == "thinking":
                        if "thinking" in item.content[0]:
                            thoughts += item.content[0]["thinking"]
                    elif item.content[0]["type"] == "text":
                        if "text" in item.content[0]:
                            final_response += item.content[0]["text"]
        time.sleep(0.05)

    return thoughts, final_response  # Retorna pensamentos como uma única string

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        thoughts, response = get_response(user_input)

        # Exibir pensamentos
        if thoughts.strip():
            print("\n🤖 Pensamentos do modelo:\n💭", thoughts)

        # Exibir resposta final
        if response.strip():
            print(f"\nBot: {response}")

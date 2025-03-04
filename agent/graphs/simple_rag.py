from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from agent.nodes.nodes import SimpleRAG
from rag.retrieval.standart_retriever import StandardRetriever
from agent.states.simple_rag_state import SimpleRagState

SimpleRAG = SimpleRAG(
    llm_provider='anthropic',
    llm_model_name='claude-3-7-sonnet-latest'
)

memory = MemorySaver()

# Build graph
builder = StateGraph(SimpleRagState)

# Add nodes
builder.add_node("assistant", SimpleRAG.assistant)
builder.add_node("retrieval", SimpleRAG.retrieval)

# Add edges
builder.add_edge(START, "retrieval")
builder.add_edge("retrieval", "assistant")
builder.add_edge("assistant", END)

# Compile graph
graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}
    messages = [HumanMessage(
        content="What is HyDE?")
    ]
    result = graph.invoke({"messages": messages}, config)

    for message in result["messages"]:
        if isinstance(message, AIMessage):
            print("=====AI Message=====")
            for msg in message.content:
                if msg["type"] == "thinking":
                    print("---Model is thinking---")
                    print(msg["thinking"])

                if msg["type"] == "text":
                    print("---Model answer---")
                    print("Model's final response:\n", msg["text"])

        if isinstance(message, HumanMessage):
            print("=====Human Message=====")
            print("User's input:\n", message.content)


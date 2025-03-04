from langgraph.prebuilt import tools_condition, ToolNode
from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from agent.nodes.nodes import SimpleRAG
from rag.retrieval.standart_retriever import StandardRetriever
from agent.states.simple_rag_state import SimpleRagState
import asyncio

SimpleRAG = SimpleRAG(
    llm_provider='anthropic',
    llm_model_name='claude-3-7-sonnet-latest'
)

memory = MemorySaver()

# Build graph
builder = StateGraph(SimpleRagState)

# Add nodes
builder.add_node("retrieval", SimpleRAG.retrieval)
builder.add_node("assistant", SimpleRAG.assistant)

# Add edges
builder.add_edge(START, "retrieval")
builder.add_edge("retrieval", "assistant")

# Compile graph
graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1"}}

    def get_response(msg):
        messages = [HumanMessage(
            content=msg)
        ]

        for msg, metadata in graph.stream(
                {"messages": messages},
                stream_mode="messages", config=config
        ):
            if msg.content:
                yield msg.content

    print(get_response("What is HyDE?"))

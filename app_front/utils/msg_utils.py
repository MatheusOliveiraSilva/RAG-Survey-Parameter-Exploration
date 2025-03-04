from agent.graphs.simple_rag import graph
from langchain_core.messages import HumanMessage
import time

def get_response(result, llm_provider) -> str:

    if llm_provider == 'anthropic':
        return result["messages"][-1].content[-1]['text']

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

def thinking_mode_response_generator(user_input, config):
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

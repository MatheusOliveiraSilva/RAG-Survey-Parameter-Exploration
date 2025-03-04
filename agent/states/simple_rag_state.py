from typing_extensions import TypedDict
from typing import Annotated, List

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from rag.retrieval.standart_retriever import StandardRetriever

class SimpleRagState(TypedDict):
    messages: Annotated[list, add_messages]
    context: List[Document]


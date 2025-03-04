from agent.states.simple_rag_state import SimpleRagState
from agent.prompts.prompts import SIMPLE_RAG_PROMPT
from configs.models_config import ModelsConfig
from langchain_core.messages import SystemMessage
from rag.retrieval.standart_retriever import StandardRetriever

class SimpleRAG:
    def __init__(self,
                 llm_provider: str,
                 llm_model_name: str
                 ) -> None:

        self.model_config = ModelsConfig(provider=llm_provider)
        self.llm_model_name = llm_model_name

        self.retriever = StandardRetriever(
            embedding_model_name="text-embedding-3-large",
            embedding_provider="openai",
            metric="dotproduct"
        )

    def retrieval(self, state: SimpleRagState):
        """
        Retrieval node
        """
        query = state["messages"][-1].content
        state["context"] = self.retriever.retrieve(query)

        return state

    def assistant(self, state: SimpleRagState):
        """
        Assistant node
        """

        llm = self.model_config.get_llm_model(
            model_name="claude-3-7-sonnet-latest",
            max_tokens=2048,
            thinking={"type": "enabled", "budget_tokens": 1024}
        )

        sys_msg = SystemMessage(
            content=SIMPLE_RAG_PROMPT.format(
                relevant_documents=state["context"],
                message=state["messages"][-1].content
            )
        )

        response = llm.invoke(
            [sys_msg] + state["messages"]
        )

        return {"response": response.content}

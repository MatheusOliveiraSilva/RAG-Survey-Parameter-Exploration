from agent.states.simple_rag_state import SimpleRagState
from agent.prompts.prompts import SIMPLE_RAG_PROMPT
from configs.models_config import ModelsConfig

class SimpleRAG:
    def __init__(self,
                 llm_provider: str,
                 llm_model_name: str,
    ) -> None:

        self.model_config = ModelsConfig(provider=llm_provider)
        self.llm_model_name = llm_model_name

    def retrieval(self, state: SimpleRagState):
        """
        Retrieval node
        """
        query = state["messages"][-1].content
        state["context"] = state["retriever"].retrieve(query)

        return state

    def assistant(self, state: SimpleRagState):
        """
        Assistant node
        """

        llm = self.model_config.get_llm_model(
            model_name="claude-3-7-sonnet-latest",
            thinking={"type": "enabled", "budget_tokens": 2000}
        )

        sys_msg = SystemMessage(
            content=SIMPLE_RAG_PROMPT.format(
                relevant_documents=state["context"]
            )
        )

        return {"messages": [llm.invoke([sys_msg] + state["messages"])]}


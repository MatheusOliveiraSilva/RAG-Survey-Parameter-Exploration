from langchain_core.documents import Document
from rag.vector_store.pinecone_utils import PineconeUtils
from typing import List

class StandardRetriever:
    def __init__(self,
                 embedding_model_name: str,
                 embedding_provider: str,
                 top_k: int = 10
                 ):

        self.top_k = top_k
        self.vector_store = PineconeUtils(
            index_name=f"rag-{embedding_model_name}",
            embedding_model_name=embedding_model_name,
            embedding_provider=embedding_provider
        ).vector_store

    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """
        This function retrieves the top k documents from the database.
        """
        return self.vector_store.similarity_search_with_score(query=query, k=top_k)

if __name__ == "__main__":
    retrieval = StandardRetriever(embedding_model_name="text-embedding-3-large", embedding_provider="openai")
    documents = retrieval.retrieve("How HyDE works?")
    print(documents)
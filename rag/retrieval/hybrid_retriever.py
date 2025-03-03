from pinecone_text.sparse import BM25Encoder
from langchain_core.documents import Document
from rag.vector_store.pinecone_utils import PineconeUtils
from rag.embedding.embedding_config import EmbeddingConfig
from langchain_community.retrievers import PineconeHybridSearchRetriever
from typing import List

class HybridSearchRetriever:
    def __init__(self,
                 embedding_model_name: str,
                 embedding_provider: str,
                 top_k: int = 10
                 ):

        self.top_k = top_k
        self.index = PineconeUtils(
            index_name=f"rag-{embedding_model_name}",
            embedding_model_name=embedding_model_name,
            embedding_provider=embedding_provider
        ).index

        self.bm25_encoder = BM25Encoder().default()
        self.embeddings = EmbeddingConfig(
            embedding_model=embedding_model_name,
            provider=embedding_provider
        ).get_embedding_model()

        self.retriever = PineconeHybridSearchRetriever(
            index=self.index, bm25_encoder=self.bm25_encoder,
            embeddings=self.embeddings
        )


    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """
        This function retrieves the top k documents from the database.
        """
        return self.vector_store.similarity_search_with_score(query=query, k=top_k)

if __name__ == "__main__":
    retriever = HybridSearchRetriever(
        embedding_model_name="text-embedding-3-large",
        embedding_provider="openai"
    ).retriever

    documents = retriever.invoke("How HyDE works?")
    print(documents)
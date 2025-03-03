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
                 ) -> None:

        self.top_k = top_k
        self.index = PineconeUtils(
            index_name=f"rag-{embedding_model_name}-dotproduct",
            metric="dotproduct",
            embedding_model_name=embedding_model_name,
            embedding_provider=embedding_provider
        ).index

        # Inicializa o BM25Encoder corretamente
        self.sparse_encoder = BM25Encoder().default()

        self.embeddings = EmbeddingConfig(
            embedding_model=embedding_model_name,
            provider=embedding_provider
        ).get_embedding_model()

        # Passa o encoder corretamente como "sparse_encoder"
        self.retriever = PineconeHybridSearchRetriever(
            index=self.index,
            sparse_encoder=self.sparse_encoder,  # Corrigido
            embeddings=self.embeddings,
            text_key="text"
        )

    def retrieve(self, query: str) -> List[Document]:
        """
        This function retrieves the top k documents from the database.
        """
        return self.retriever.invoke(query)

if __name__ == "__main__":
    retriever = HybridSearchRetriever(
        embedding_model_name="text-embedding-3-large",
        embedding_provider="openai"
    ).retriever

    documents = retriever.invoke("How HyDE works?")
    print(documents)

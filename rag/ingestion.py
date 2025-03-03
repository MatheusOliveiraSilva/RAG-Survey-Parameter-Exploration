from rag.chunking.chunking_config import ChunkingConfig
from vector_store.pinecone_utils import PineconeUtils
from langchain_community.document_loaders import DirectoryLoader
from pathlib import Path

class Ingestion:
    def __init__(self,
                 chunk_size, chunk_overlap, chunking_strategy,
                 directory, embedding_model_name,
                 model_provider: str ='openai',
                 glob: str = '*.pdf'):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.chunker = self.define_chunker()

        self.docs = self.load_documents(directory, glob)
        self.chunked_docs = self.chunker.split_documents(self.docs)

        self.pinecone_utils = PineconeUtils(
            index_name=f'rag-{embedding_model_name}',
            embedding_model_name=embedding_model_name,
            embedding_provider=model_provider
        )

        self.pinecone_utils.insert_documents(self.chunked_docs)

    def define_chunker(self):
        chunking_config = ChunkingConfig(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            chunking_strategy=self.chunking_strategy
        )

        return chunking_config.get_chunker()

    def load_documents(self, directory, glob):
        loader = DirectoryLoader(
            directory, glob=glob, show_progress=True
        )

        return loader.load()

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent
    path = ROOT_DIR / 'documents'

    ingestion = Ingestion(
        chunk_size=512,
        chunk_overlap=50,
        chunking_strategy='fixed-size-chunking',
        directory=path,
        embedding_model_name='text-embedding-3-large'
    )
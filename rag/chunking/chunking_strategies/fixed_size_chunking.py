from langchain_text_splitters import CharacterTextSplitter

class FixedSizeChunking:
    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 80)\
            -> None:
        """
        Class for fixed size chunking strategy.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def get_chunker(self):
        return CharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
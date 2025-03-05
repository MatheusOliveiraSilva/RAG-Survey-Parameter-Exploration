import os
from rag.chunking.chunking_strategies.fixed_size_chunking import FixedSizeChunking
from rag.chunking.chunking_strategies.content_aware_chunking import DynamicSizeChunking

class ChunkingConfig:
    def __init__(self,
                 chunk_size: int,
                 chunk_overlap: int,
                 chunking_strategy: str,
                 chunking_strategy_params: dict = None):
        """
        Class used to manage with chunking configuration we are using in the project.
        """

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.chunking_strategy_params = chunking_strategy_params

    def get_chunker(self):

        if self.chunking_strategy == 'fixed-size':
            return FixedSizeChunking(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            ).get_chunker()

        if self.chunking_strategy == 'content-aware':
            return DynamicSizeChunking(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            ).get_chunker()

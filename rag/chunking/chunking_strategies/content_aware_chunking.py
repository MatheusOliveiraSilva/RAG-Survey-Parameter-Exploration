from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Literal

class DynamicSizeChunking:
    def __init__(self,
                 chunk_size: int = 512,
                 chunk_overlap: int = 80,
                 document_type: Literal["PDF", "HTML", "Markdown", "JSON", "Code"] = "PDF")\
            -> None:
        """
        Class for content aware chunking strategy.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.document_type = document_type

    def get_chunker(self):
        if self.document_type == "PDF":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

        # TODO: Implement other document types

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=ROOT_DIR / '.env')

class EmbeddingConfig:
    def __init__(self,
                 embedding_model: str,
                 provider: str)\
            -> None:
        """
        Class used to manage with embedding configuration we are using in the project.
        """
        self.provider = provider
        self.instanced_model = self.instance_embedding_model(embedding_model)
        self.embedding_model_name = embedding_model

    def instance_embedding_model(self, embedding_model: str):

        if self.provider == "azure":
            instanced_model = AzureOpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                openai_api_version=os.getenv("OPENAI_API_VERSION"),
                openai_api_base=os.getenv("AZURE_OPENAI_BASE_URL")
            )
        if self.provider == "openai":
            instanced_model = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=os.getenv("OLD_OPENAI_API_KEY")
            )
        return instanced_model

    def get_embedding_model(self):
        return self.instanced_model

if __name__ == "__main__":
    embedding_config = EmbeddingConfig(embedding_model="text-embedding-ada-002")
    embedding_model = embedding_config.get_embedding_model()
    print(embedding_config.embedding_model_name)
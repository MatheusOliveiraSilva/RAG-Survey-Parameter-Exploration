import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from configs.models_config import ModelsConfig
from tqdm import tqdm
from anthropic import RateLimitError  # Certifique-se de que esse import está correto

class Preprocesser:
    def __init__(self, preprocessing_technique: str) -> None:
        self.preprocessing_technique = preprocessing_technique
        llm_config = ModelsConfig('anthropic')
        self.llm = llm_config.get_llm_model(
            model_name="claude-3-7-sonnet-latest"
        )

    def contextual_embedding(self, docs):
        """
        Técnica de contextual embedding.
        """
        for doc in tqdm(docs, desc="Augmenting chunks using contextual embeddings strategy..."):
            source_doc_path = doc.metadata['source']
            loader = PyPDFLoader(source_doc_path)
            source_doc_pages = loader.load()

            full_source_document = ""
            for page in source_doc_pages:
                full_source_document += page.page_content + "\n"

            sys_msg = f"""
Considering the full document:
<document>
{full_source_document}
</document>
"""
            human_msg = f"""
Here is the chunk we want to situate within the whole document:
<chunk>
{doc.page_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=[{
                    "text": sys_msg,
                    "type": "text",
                    "cache_control": {"type": "ephemeral"}
                }]),
                HumanMessage(content=human_msg)
            ])

            # try catch because of api limit error (rate limit)
            try:
                result = self.llm.invoke(prompt.messages)
            except RateLimitError:
                print("\nRate limit error. Waiting 60 sec before trying again...")
                time.sleep(60)
                try:
                    result = self.llm.invoke(prompt.messages)
                except RateLimitError as e:
                    print("Rate limit persists. Skipping this chunk. \n", e)
                    continue

            # Adiciona a resposta ao final do conteúdo do documento
            doc.page_content += "\n" + result.content

        return docs

    def preprocess_documents(self, docs):
        if self.preprocessing_technique == "contextual-embedding":
            return self.contextual_embedding(docs)
        return docs


if __name__ == "__main__":

    docs = [
        Document(
            metadata={'source': '/Users/matheus/Documents/pessoal/RAG-Survey-Parameter-Exploration/documents/hyde.pdf'},
            page_content='2 2 0 2\n\nc e D 0 2\n\n]\n\nR\n\nI . s c [\n\n1 v 6 9 4 0 1 . 2 1 2 2 : v i X r a\n\nPrecise Zero-Shot Dense Retrieval without Relevance Labels\n\nLuyu Gao∗ † Xueguang Ma∗ ‡\n\nJimmy Lin‡\n\nJamie Callan†\n\n†Language Technologies Institute, Carnegie Mellon University ‡David R. Cheriton School of Computer Science, University of Waterloo {luyug, callan}@cs.cmu.edu, {x93ma, jimmylin}@uwaterloo.ca\n\nAbstract'
        )
    ]

    preprocessor = Preprocesser(preprocessing_technique="contextual-embedding")
    print(preprocessor.preprocess_documents(docs=docs))

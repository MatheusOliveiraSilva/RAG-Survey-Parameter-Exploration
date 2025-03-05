from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from configs.models_config import ModelsConfig

class Preprocesser:
    def __init__(self,
                 preprocessing_technique: str
                 ) -> None:

        self.preprocessing_technique = preprocessing_technique

        llm_config = ModelsConfig('anthropic')

        self.llm = llm_config.get_llm_model(
            model_name="claude-3-7-sonnet-latest"
        )

    def contextual_embedding(self, docs):
        """
        Contextual embedding technique
        """
        for doc in docs:
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
                SystemMessage(content=[
                    {
                        "text": sys_msg,
                        "type": "text",
                        "cache_control": {"type": "ephemeral"},
                    }
                ]),
                HumanMessage(content=human_msg)
            ])

            result = self.llm.invoke(prompt.messages)
            doc.page_content = result.content

        return docs

    def preprocess_documents(self, docs):
        if self.preprocessing_technique == "contextual-embedding":
            return self.contextual_embedding(docs)

        return docs


if __name__ == "__main__":

    docs = [Document(metadata={'source': '/Users/matheus/Documents/pessoal/RAG-Survey-Parameter-Exploration/documents/hyde.pdf'}, page_content='2 2 0 2\n\nc e D 0 2\n\n]\n\nR\n\nI . s c [\n\n1 v 6 9 4 0 1 . 2 1 2 2 : v i X r a\n\nPrecise Zero-Shot Dense Retrieval without Relevance Labels\n\nLuyu Gao∗ † Xueguang Ma∗ ‡\n\nJimmy Lin‡\n\nJamie Callan†\n\n†Language Technologies Institute, Carnegie Mellon University ‡David R. Cheriton School of Computer Science, University of Waterloo {luyug, callan}@cs.cmu.edu, {x93ma, jimmylin}@uwaterloo.ca\n\nAbstract'), Document(metadata={'source': '/Users/matheus/Documents/pessoal/RAG-Survey-Parameter-Exploration/documents/hyde.pdf'}, page_content='While dense retrieval has been shown effec- tive and efﬁcient across tasks and languages, it remains difﬁcult to create effective fully zero-shot dense retrieval systems when no rel- evance label is available. In this paper, we recognize the difﬁculty of zero-shot learning and encoding relevance. Instead, we pro- pose to pivot through Hypothetical Document Embeddings (HyDE). Given a query, HyDE ﬁrst zero-shot instructs an instruction-following language model (e.g. InstructGPT) to gen- erate a hypothetical document. The docu- ment captures relevance patterns but is unreal and may contain false details. Then, an un- supervised contrastively learned encoder (e.g. Contriever) encodes the document into an embedding vector. This vector identiﬁes a neighborhood in the corpus embedding space, where similar real documents are retrieved based on vector similarity. This second step ground the generated document to the actual corpus, with the encoder’s dense bottleneck ﬁltering out the incorrect details. Our exper- iments show that HyDE signiﬁcantly outper- forms the state-of-the-art unsupervised dense retriever Contriever and shows strong per- formance comparable to ﬁne-tuned retrievers, across various tasks (e.g. web search, QA, fact veriﬁcation) and languages (e.g. sw, ko, ja).1')]

    preprocessor = Preprocesser(preprocessing_technique="contextual-embedding")
    print(preprocessor.preprocess_documents(docs=docs))

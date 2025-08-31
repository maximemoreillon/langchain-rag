from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


def createVectorStore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return InMemoryVectorStore(embeddings)


vector_store = createVectorStore()

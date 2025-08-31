from os import getenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings


PINECONE_API_KEY = getenv("PINECONE_API_KEY")
PROXY_URL = getenv("HTTPS_PRROXY")
pc = Pinecone(
    api_key=PINECONE_API_KEY,
    proxy_url=PROXY_URL,
)


def createVectorStore():
    index_name = "langchain-rag"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create index if not exists
    if not pc.has_index(index_name):
        print(f"Index {index_name} does not exist. Creating...")
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"Index {index_name} created")

    index = pc.Index(index_name)

    return PineconeVectorStore(index=index, embedding=embeddings)


vector_store = createVectorStore()

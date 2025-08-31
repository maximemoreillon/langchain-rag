from dotenv import load_dotenv

load_dotenv()

from inMemory.vectorStore import vector_store as inMemoryVectorStore
from pineconeDb.vectorStore import vector_store as pineConeVectorStore

# from documentSraping.web import getDocuments
from documentSraping.local import getDocuments

from langchain_text_splitters import RecursiveCharacterTextSplitter

vector_store = pineConeVectorStore


def splitDocs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    print(f"Split documents into {len(all_splits)} sub-documents.")

    return all_splits


docs = getDocuments()
splits = splitDocs(docs)

vector_store.add_documents(documents=splits)

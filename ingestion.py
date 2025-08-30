from typing_extensions import List, TypedDict
from langchain_community.document_loaders import WebBaseLoader
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import bs4


def createVectorStore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return InMemoryVectorStore(embeddings)


def scrapePage():
    # Only keep post title, headers, and content from the full HTML.
    url = "https://articles.maximemoreillon.com/articles/277"
    bs4_strainer = bs4.SoupStrainer()
    loader = WebBaseLoader(
        web_paths=(url),
        bs_kwargs={"parse_only": bs4_strainer},
    )
    docs = loader.load()

    assert len(docs) == 1
    print(f"Total characters: {len(docs[0].page_content)}")

    return docs


def splitDocs(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    print(f"Split blog post into {len(all_splits)} sub-documents.")

    return all_splits


def store(all_splits):
    vector_store.add_documents(documents=all_splits)  # Index chunks


def ingest():
    docs = scrapePage()
    splits = splitDocs(docs)
    store(splits)


vector_store = createVectorStore()

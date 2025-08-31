from dotenv import load_dotenv

load_dotenv()

from typing_extensions import List, TypedDict
from langchain import hub
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langchain_openai import ChatOpenAI
from inMemory.vectorStore import vector_store as inMemoryVectorStore
from pineconeDb.vectorStore import vector_store as pineconeVectorStore

llm = ChatOpenAI(model="gpt-4o-mini")
vector_store = pineconeVectorStore


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):

    # Define prompt for question-answering
    # Prompt contains template among others
    prompt = hub.pull("rlm/rag-prompt")

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


def makeGraph():

    graph_builder = StateGraph(State).add_sequence([retrieve, generate])

    # Make the "retrieve" edge the start point
    graph_builder.add_edge(START, "retrieve")

    return graph_builder.compile()


def main():
    question = "How is the stock market doing today?"
    graph = makeGraph()
    response = graph.invoke({"question": question})
    print(response["answer"])


if __name__ == "__main__":
    main()

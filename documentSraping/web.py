from langchain_community.document_loaders import WebBaseLoader
import bs4


def getDocuments():
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

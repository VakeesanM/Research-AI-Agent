from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader


@tool
def content_extractor(url: str):
    """
    Fetch and extract the full text content from a given URL.

    Use this tool when you have a URL and need to read the actual contents of the
    page. Returns a document object containing the extracted text.

    Args:
        url (str): The full URL of the webpage to extract content from.

    Returns:
        Document: A document object containing the page's extracted text content.
    """
    content = WebBaseLoader(url).load()
    pages = ""
    for doc in content:
        pages = pages + doc.page_content
    return pages
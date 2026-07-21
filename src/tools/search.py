from ddgs import DDGS
from langchain_core.tools import tool


@tool
def search(topic: str):
    """
    Search the web for sources relevant to a given research topic.

    Use this tool when you need to discover websites, articles, or pages related
    to a topic. Returns a dictionary mapping source titles to their URLs.

    Args:
        topic (str): The research topic or query to search for.

    Returns:
        dict[str, str]: A dictionary where keys are source titles and values are URLs.

    Example:
        {"MIT Technology Review": "https://www.technologyreview.com/..."}
    """
    urls = {}
    with DDGS() as ddgs:
        results = ddgs.text(topic, max_results=5)
        for r in results:
            urls[r['title']] = r['href']
    return urls
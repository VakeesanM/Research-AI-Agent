from typing import List
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from tools.summary import summarizer
import asyncio

@tool
def content_extractor(urls: List[str]):
    """
    Fetch and extract the full text content from a given URL.

    Use this tool when you have URLs and need to read the actual contents of the
    page. Returns a summary of the extracted text. 

    Args:
        url (List[str]): A list of full URLs of the webpages to extract content from.

    Returns:
        Summary (List[str]): A list of document summaries of each URL's texts.
        
    """
    return asyncio.run(content_extractor_helper(urls))
async def content_extractor_helper(urls):
    return await asyncio.gather(*(extract_and_summarize(u) for u in urls))


async def extract_and_summarize(url):
    content = await asyncio.to_thread(WebBaseLoader(url).load)
    pages = ''
    for doc in content:
            pages = pages + doc.page_content
    summary = await summarizer(pages)
    return summary


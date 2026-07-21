from langchain_community.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


summary_prompt = ChatPromptTemplate([
    ("system", """You are a summarization tool within a ReAct Agent pipeline.
     
    Your task: Given an article, produce a concise, clear summary that captures the key ideas without unnecessary detail.

    Guidelines:
    - Lead with the article's main thesis or purpose
    - Include only the most important supporting points
    - Avoid filler phrases and redundancy
    - Write in plain, direct language"""),
    ("human", "Article Content:\n\n{content}")
])

model = ChatOpenAI(model="gpt-4o-mini")
summary_chain = summary_prompt|model


@tool
def summarizer(document: str):
    """
    Summarize the key information from a document.

    Use this tool after extracting page content when you need a concise overview
    rather than the full text. Useful for processing multiple sources efficiently.

    Args:
        document (Document): The document object to summarize.

    Returns:
        str: A concise summary of the document's main points.
    """
    return summary_chain.invoke({"content": document}).content
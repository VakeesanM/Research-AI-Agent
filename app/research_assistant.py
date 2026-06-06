from langgraph.graph import StateGraph, START, END
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, ToolMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Sequence
from ddgs import DDGS
import arxiv
from dotenv import load_dotenv
import os
import wikipediaapi


load_dotenv()
os.getenv("OPENAI_API_KEY")




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

wiki = wikipediaapi.Wikipedia(
    user_agent="MyApp/1.0 (RAA@email.com)",  
    language="en"
)

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

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

@tool
def getabstract(topic: str, amount=5):
    """
    Find and summarize academic papers related to a research topic.

    Use this tool when you need peer-reviewed or scholarly sources. Returns
    summaries of the 5 most relevant academic papers found for the topic.

    Args:
        topic (str): The research topic to find academic papers for.
        amount(int): The amount of research papers you want. Default is 5. Don't use more than 10.  

    Returns:
        list[dict]: A list of 5 papers, each containing title, authors, and summary.
    """
    client = arxiv.Client()
    try: 
        search = arxiv.Search(
            query=topic,
            max_results=amount,
            sort_by=arxiv.SortCriterion.Relevance  
            
        )
        results = client.results(search)

        papers = []
        for paper in results:
            info = [paper.title, paper.authors, paper.summary, paper.entry_id]
            papers.append(info)
    except Exception as e:
        return e
    else:
        return papers

@tool
def searchIdeas(topic:str):
    """
    Break a broad research topic down into more specific subtopics.

    Use this tool when a topic is too vague or general to research effectively.
    Returns a list of narrower, more concrete angles to explore instead.

    Args:
        topic (str): A broad or abstract topic to narrow down.

    Returns:
        list[str]: A list of specific subtopics derived from the original topic.

    Example:
        Input:  "python"
        Output: ["Python Programming", "Pythonidae", "Monty Python"]
    """
    results = wiki.search(topic)
    return list(results.pages.keys())

tools = [search, content_extractor, summarizer, getabstract, searchIdeas]

agent_model = ChatOpenAI(model="gpt-4o-mini", streaming=True).bind_tools(tools=tools)


def agent_call(state: State) -> State:
    system_message = SystemMessage("""You are an AI Research Assistant. Your purpose is to generate a comprehensive research brief on a given topic.

    ## Behavior
    - If the topic is too vague, use the `searchIdeas` tool to retrieve a list of more specific ideas. Choose the most appropriate and interesting one to research.
    - When searching for articles or academic papers, always use specific, targeted search queries rather than broad ones.
    - If a tool returns a error for a parameter. Don't retry that tool with a same parameter. 
    - The content_extractor tool will most likely return errors for certain urls, ignore those urls and try other one.
    - If you get a 429 error, stop fetching for any more sources at the moment and create the breif with only the given summaries/abstracts. 

    ## Research Process
    1. Gather sources using your available tools — look up both articles and academic papers.
    2. Note that article tools return full content, while paper tools return abstracts only.
    3. Summarize each article's content before incorporating it into the brief.
    4. Once you have summaries/abstract from atleast 5-6 sources, consisting of mostly academic papers and some articles, compile the research brief.

    ## Output Format
    Respond only with the research brief in the following structure:

    # [Topic Title]

    ## Main Idea
    A single paragraph summarizing the core finding or thesis.

    ## Key Ideas
    - **[Idea Title]**: A 2–3 sentence explanation of this idea.
    - **[Idea Title]**: A 2–3 sentence explanation of this idea.
    - *(continue for all key ideas)*

    ## Sources
    | Title | Type | Link |
    |-------|------|------|
    | ...   | Academic Paper / Article | ... |
    """)
    respone = agent_model.invoke([system_message] + state['messages'])
    return {"messages": [respone]}

def shouldEnd(state:State):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "finish"
    else:
        return "Use Tools"
    
graph = StateGraph(State)

graph.add_node("agent", agent_call)
graph.add_node("tools", ToolNode(tools=tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    shouldEnd,
    {
        "finish": END,
        "Use Tools": "tools"
    }
)

graph.add_edge("tools", "agent")

app = graph.compile()

image = app.get_graph().draw_mermaid_png()

def get_brief(topic: str):
    result = app.invoke({
    "messages": [HumanMessage(topic)]
    })
    return result['messages'][-1].content

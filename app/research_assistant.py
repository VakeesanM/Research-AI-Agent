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
    """This fucntion takes in a topic and searchs for relevant websites for that topic. It returns a python dictonary of websites and their urls."""
    urls = {}
    with DDGS() as ddgs:
        results = ddgs.text(topic, max_results=5)
        for r in results:
            urls[r['title']] = r['href']
    return urls
@tool
def content_extractor(url: str):
    """This function extracts the content of the page found at given url and returns a document containing the url's contents"""
    content = WebBaseLoader(url).load()
    pages = ""
    for doc in content:
        pages = pages + doc.page_content
    return pages

@tool
def summarizer(document: str):
    """This function takes in a document and returns a summary of the document"""
    return summary_chain.invoke({"content": document})

@tool
def getabstract(topic: str):
    """This function finds 5 acadmic papers about the given topic and returns their summary"""
    client = arxiv.Client()

    search = arxiv.Search(
        query=topic,
        max_results=5,
        sort_by=arxiv.SortCriterion.Relevance  
        
    )
    results = client.results(search)

    papers = []
    for paper in results:
        info = [paper.title, paper.authors, paper.summary, paper.entry_id]
        papers.append(info)
    
    return papers

@tool
def searchIdeas(topic:str):
    """Given a topic, it returns a list of topics that are more Specific and less abstract. 
        Use this when the topic you are researching is too vauge, and you want find a more specific topics"""
    results = wiki.search(topic)
    return results.pages.keys()

tools = [search, content_extractor, summarizer, getabstract, searchIdeas]

agent_model = ChatOpenAI(model="gpt-4o-mini", streaming=True).bind_tools(tools=tools)


def agent_call(state: State) -> State:
    system_message = SystemMessage("""You are an AI Research Assistant. Your purpose is to generate a comprehensive research brief on a given topic.

    ## Behavior
    - If the topic is too vague, use the `searchIdeas` tool to retrieve a list of more specific ideas. Choose the most appropriate and interesting one to research.
    - When searching for articles or academic papers, always use specific, targeted search queries rather than broad ones.

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

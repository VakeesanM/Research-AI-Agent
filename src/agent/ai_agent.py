from tools.abstract import get_abstract
from tools.content import content_extractor
from tools.ideas import search_ideas
from tools.search import search
from tools.summary import summarizer
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from agent.state import State


from agent.end import should_end
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


class Agent():
    def __init__(self):
        self.tools = [get_abstract, content_extractor, search_ideas, search, summarizer]
        self.main_model = ChatOpenAI(model="gpt-4o-mini", streaming=True).bind_tools(tools=self.tools)
        self.agent = self.create_agent()


    
    def create_agent(self):
        graph = StateGraph(State)

        graph.add_node("agent", self.start_agent)
        graph.add_node("tools", ToolNode(tools=self.tools))

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            should_end,
            {
                "finish": END,
                "Use Tools": "tools"
            }
        )

        graph.add_edge("tools", "agent")

        app = graph.compile()

        return app

    def start_agent(self, state:State):
        system_message = SystemMessage("""
            You are an AI Research Assistant. Your purpose is to generate a comprehensive research brief on a given topic.

            ## Behavior
            - If the topic is too vague, use the `searchIdeas` tool to retrieve a list of more specific ideas. Choose the most appropriate and interesting one to research.
            - When searching for articles or academic papers, always use specific, targeted search queries rather than broad ones.
            - If a tool returns a error for a parameter. Don't retry that tool with a same parameter. 
            - The content_extractor tool will most likely return errors for certain urls, ignore those urls and try other one.
            - If you get a error from the getabstract, stop fetching for any more sources at the moment and create the breif with only the given summaries/abstracts. 
            - Both the getabstract and search return various abstracts and articles, repestively. Use only articles/abstracts that are relevant to the topic at hand. 

            ## Research Process
            1. Gather sources using your available tools — look up both articles and academic papers.
            2. Note that article tools return full content, while paper tools return abstracts only.
            3. Summarize each article's content before incorporating it into the brief.
            4. Once you have summaries/abstract from atleast 2-6 sources, consisting of academic papers and articles relevant to the topic, compile the research brief.
                                       
            ## Important
            1. Don't include irrevelant articles and papers in the sources section.
            2. Verify that all links are active and functional.

            ## Output Format
            Respond only with the research brief in the following structure:

            # [Topic Title]

            ## Main Idea
            A single paragraph summarizing the core finding or thesis.

            ## Key Ideas
            - **[Idea Title]**: A 2–3 sentence explanation of this idea that relates to the main topic.
            - **[Idea Title]**: A 2–3 sentence explanation of this idea that relates to the main topic.
            - *(continue for all key ideas)*

            ## Sources
            | Title | Type | Link |
            |-------|------|------|
            | ...   | Academic Paper / Article | ... |
            """)
        respone = self.main_model.invoke([system_message] + state['messages'])
        return {"messages": [respone]}

    def get_brief(self, topic:str):
        result = self.agent.invoke({
        "messages": [HumanMessage(topic)]
        }, config={"recursion_limit": 25})
        return result['messages'][-1].content


from langchain_community.tools import tool
import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    user_agent="MyApp/1.0 (RAA@email.com)",  
    language="en"
)


@tool
def search_ideas(topic:str):
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

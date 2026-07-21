from langchain_community.tools import tool
import arxiv

@tool
def get_abstract(topic: str, amount=5):
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
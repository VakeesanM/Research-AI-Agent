## What I Built

I built an Autonomous ReAct AI Assistant capable of retrieving articles and their summaries for topics chosen by the user. This agent's main purpose is to give people a starting point for their research by providing links to relevant articles and academic papers along with their summaries.

Link to Website: [Click Here](https://huggingface.co/spaces/VakeesanM/Research-Assistant-Agent)
## How It Works

This is an Autonomous ReAct Agent that uses its own logic to create a workflow. It uses tools bound to it to retrieve articles and academic papers, generate summaries, and finally output a research brief alongside article and academic paper URLs.

![Structure of ReAct Agent](agent.png)

## Tools & Tech Stack

The tools used by the AI Assistant are:
* **Search** - Using DuckDuckGo's API, the assistant looks up and retrieves URLs for relevant websites.
* **Content Extractor** - Allows the assistant to extract the contents of a page given a URL.
* **Summarizer** - Given article content, returns a summary of it.
* **getabstract** - Using arXiv's API, the assistant can retrieve abstracts from arXiv.
* **searchIdeas** - Using Wikipedia's API, the assistant can look up topics related to the user's topic. Note: this is only used when the topic given by the user is too vague and a more specific version can be retrieved using Wikipedia.

The tech stack used to build this AI Assistant consists of:
* Langchain
* Langchain Community
* Langchain OpenAI
* Langgraph
* Wikipedia API
* ArXiv API
* DuckDuckGo API

## How to Run Locally
Requires OpenAI API Key
```bash
git clone "https://github.com/VakeesanM/Research-AI-Agent.git"
pip install -r requirements.txt
streamlit run "app/app.py"
```

## Challenges & How I Solved Them

* DuckDuckGo's API only returns URLs, so I built a custom content extractor function to fetch and parse the actual page content.
* The agent struggled with vague topics, returning poor results — adding a Wikipedia search tool allowed it to identify more specific, researchable variations of a topic before proceeding.
* Early testing revealed the agent was searching topics too literally instead of exploring them in depth. I addressed this by prompting the agent to generate more complex, targeted questions around the topic before beginning its research.
* Certain websites would block attempts to extact content from them. I fixed this by instructing the Agent to not repeat any urls thats return errors and adding try catch blocks . 
* Set Recursion Limit, which reduced run time without causing any serious loss in quality of results. 
# Research Assistant Agent

An autonomous ReAct AI agent that retrieves relevant articles and academic papers on any topic and generates concise summaries — giving researchers a fast, reliable starting point instead of manually digging through search results.

🔗 **Live Demo:** [Try it here](https://huggingface.co/spaces/VakeesanM/Research-Assistant-Agent)

---

## How It Works

The agent follows the ReAct (Reason + Act) pattern, autonomously deciding which tools to call and in what order. Given a topic, it plans its own research workflow — searching the web and arXiv, extracting and summarizing content — and returns a research brief with links to every source it used.

![Structure of ReAct Agent](agent.png)

---

## Tools & Tech Stack

**Agent Tools**

| Tool | Description |
|------|-------------|
| **Search** | Uses DuckDuckGo's API to find relevant URLs for a given query |
| **Content Extractor** | Fetches and parses the full text content of a webpage from its URL |
| **Summarizer** | Condenses extracted article content into a short summary |
| **getAbstract** | Uses arXiv's API to retrieve abstracts for relevant academic papers |
| **searchIdeas** | Uses Wikipedia's API to surface more specific, researchable subtopics when the user's query is too broad or vague |

**Tech Stack**
- LangChain / LangChain Community / LangChain OpenAI
- LangGraph
- Wikipedia API
- arXiv API
- DuckDuckGo API

---

## Running Locally

**Requirements:** OpenAI API key

```bash
git clone https://github.com/VakeesanM/Research-AI-Agent.git
cd Research-AI-Agent
pip install -r requirements.txt
#Create Env file with OpenAI apikey
streamlit run src/main.py
```

---

## Challenges & Solutions

- **DuckDuckGo only returns URLs, not content** — Built a custom content extractor to fetch and parse the actual page text.
- **Vague topics produced poor results** — Added a Wikipedia-powered `searchIdeas` tool to help the agent narrow broad queries into specific, researchable subtopics before searching.
- **Agent searched topics too literally** — Prompted the agent to generate deeper, more targeted questions around the topic before starting research, rather than searching the raw query.
- **Some sites blocked content extraction** — Added try/catch error handling and instructed the agent to skip URLs that returned errors instead of retrying them.
- **Long run times** — Set a recursion limit on the agent loop, cutting runtime with no meaningful drop in result quality.

## Adding Async

A few months after posting this project, I learned async and added it to improve latency. Async was applied to the content extraction and summarization functions, allowing multiple URLs to be extracted and summarized in parallel.

To measure the impact, I ran a small eval dataset before and after adding async. The raw results (in seconds):

| Prompt | Normal Time (s) | Async Time (s) |
|---|---|---|
| Effects of daily Omega-3 consumption | 19.07 | 8.95 |
| Custom loss functions in neural networks | 9.89 | 8.13 |
| ML in networking | 15.41 | 11.52 |
| Exercising to failure and muscle growth | 11.39 | 10.44 |
| Quantum computers' effects on computation | 10.45 | 8.57 |
| **Average** | **13.24** | **9.52** |

### Improvement in Latency

(13.24 − 9.52) / 13.24 ≈ **28.1%**

Adding async reduced average latency by roughly 28%.
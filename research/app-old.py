"""
LangChain ReAct research agent.

This app follows the working setup in agent_demo.ipynb:
- Load OPENAI_API_KEY and TAVILY_API_KEY from the project .env files.
- Use TavilySearch for web search.
- Use ChatOpenAI with the Responses API.
- Pull the ReAct prompt through LangSmith Client.
- Build and run a LangChain classic ReAct agent.

The current notebook version contains Tavily search only. It does not
contain a working weather tool, so this app does not require a weather API key.
"""

import os
from pathlib import Path

import certifi
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_classic.agents import create_react_agent, AgentExecutor
from langsmith import Client


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

# Use certifi's CA bundle for HTTPS requests, as in the notebook.
os.environ["SSL_CERT_FILE"] = certifi.where()

SCRIPT_DIR = Path(__file__).resolve().parent

# app.py is normally in C:\\Langchain-Agent\\research.
# If it is run from the project root instead, use that root.
PROJECT_DIR = (
    SCRIPT_DIR.parent
    if SCRIPT_DIR.name.lower() == "research"
    else SCRIPT_DIR
)

ENV_CANDIDATES = [
    PROJECT_DIR / ".env",
    PROJECT_DIR / "research" / ".env",
]

env_path = next((path for path in ENV_CANDIDATES if path.is_file()), None)

if env_path is None:
    checked = "\\n".join(str(path) for path in ENV_CANDIDATES)
    raise FileNotFoundError(
        "Could not find a .env file. Checked these locations:\\n" + checked
    )

# Match the notebook: load the selected .env and allow it to override
# previously loaded environment values.
load_dotenv(dotenv_path=env_path, override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()

print(f"Environment file loaded: {env_path}")
print("OpenAI API key loaded:", bool(OPENAI_API_KEY))
print("Tavily API key loaded:", bool(TAVILY_API_KEY))

missing_keys = [
    name
    for name, value in (
        ("OPENAI_API_KEY", OPENAI_API_KEY),
        ("TAVILY_API_KEY", TAVILY_API_KEY),
    )
    if not value
]

if missing_keys:
    raise RuntimeError(
        "Missing required key(s): "
        + ", ".join(missing_keys)
        + ". Please add them to the .env file."
    )


# ============================================================
# TAVILY SEARCH TOOL
# ============================================================

search_tool = TavilySearch(
    max_results=2,
)


# ============================================================
# LARGE LANGUAGE MODEL
# ============================================================

# This matches the model and Responses API setting in your notebook.
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-6-luna"),
    api_key=OPENAI_API_KEY,
    use_responses_api=True,
)


# ============================================================
# PULL REACT PROMPT FROM LANGSMITH
# ============================================================

# This follows notebook cell 7. The prompt is public, so the
# dangerously_pull_public_prompt argument is set as in the notebook.
client = Client()

prompt = client.pull_prompt(
    "hwchase17/react",
    dangerously_pull_public_prompt=True,
)


# ============================================================
# CREATE AGENT
# ============================================================

tools = [search_tool]

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)


# ============================================================
# CREATE AGENT EXECUTOR
# ============================================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
)


# ============================================================
# RUN THE AGENT
# ============================================================

def run_react_agent(question: str) -> str:
    """Run the LangChain ReAct agent; it decides when to call Tavily."""
    response = agent_executor.invoke({"input": question})
    return response.get("output", str(response))


def run_search_then_answer(question: str) -> str:
    """
    Follow the notebook's explicit three-step flow:
    user prompt -> Tavily search results -> LLM answer.
    """
    print("\\n[1/2] Searching the web with Tavily...\\n")
    search_results = search_tool.invoke(question)

    print("[2/2] Asking the LLM to analyse the search results...\\n")
    answer = llm.invoke(
        f"""
You are a careful research assistant.

Answer the user's question using the search results below.
Summarise the main relevant findings.
Include publication dates and source URLs where available.
Do not invent facts or claim that undated results are the latest.
If the search results do not contain enough information, say so.

User question:
{question}

Tavily search results:
{search_results}
"""
    )
    return answer.content


def main() -> None:
    question = input(
        "\\nWhat would you like me to research? "
        "(Press Enter for the sample Iran news query)\\n> "
    ).strip()

    if not question:
        question = "Tell me the latest news about Iran."

    print("\\nRunning Tavily search followed by LLM analysis...\\n")
    final_answer = run_search_then_answer(question)

    print("\\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60 + "\\n")
    print(final_answer)


if __name__ == "__main__":
    main()

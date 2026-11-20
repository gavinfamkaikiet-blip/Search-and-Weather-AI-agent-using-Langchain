"""
Agentic AI Assistant
Streamlit UI + LangChain ReAct agent + Tavily web search + Weatherstack current weather.

Based on the user's working agent_demo.ipynb and the instructor's Streamlit screenshots.

Expected .env variables:
    OPENAI_API_KEY=...
    TAVILY_API_KEY=...
    WEATHERSTACK_API_KEY=...
    OPENAI_MODEL=gpt-6-luna   # optional; defaults to the model used in the notebook

The API keys are loaded from .env and are never displayed in the interface.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import certifi
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_classic.agents import AgentExecutor, create_react_agent

# LangSmith is used by the original notebook to retrieve its public ReAct prompt.
# The app also has a local fallback prompt in case the prompt cannot be downloaded.
try:
    from langsmith import Client
except ImportError:
    Client = None


# ============================================================
# ENVIRONMENT AND API KEYS
# ============================================================

os.environ["SSL_CERT_FILE"] = certifi.where()

SCRIPT_DIR = Path(__file__).resolve().parent


def load_project_env() -> Path | None:
    """
    Find .env whether app.py is saved in the project root or research folder.

    Preferred locations:
      - C:\\Langchain-Agent\\research\\.env
      - C:\\Langchain-Agent\\.env
    """
    if SCRIPT_DIR.name.lower() == "research":
        candidates = [
            SCRIPT_DIR / ".env",
            SCRIPT_DIR.parent / ".env",
        ]
    else:
        candidates = [
            SCRIPT_DIR / "research" / ".env",
            SCRIPT_DIR / ".env",
        ]

    for env_file in candidates:
        if env_file.is_file():
            load_dotenv(dotenv_path=env_file, override=True)
            return env_file

    # Also allow environment variables configured outside a .env file.
    load_dotenv(override=False)
    return None


ENV_PATH = load_project_env()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY", "").strip()

# This is the model used in the supplied working notebook.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna").strip() or "gpt-6-luna"


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Agentic AI Assistant")
st.markdown(
    "Search + Weather AI Agent using **LangChain**, **Tavily** and **Weatherstack**."
)
st.caption(
    "Ask a question that needs web research, current weather information, or both."
)
st.divider()


# ============================================================
# WEATHERSTACK TOOL
# ============================================================

def create_weather_tool(api_key: str):
    """Create a LangChain tool that retrieves current weather from Weatherstack."""

    @tool("get_weather_data")
    def get_weather_data(city: str) -> str:
        """
        Fetch current temperature, weather description, humidity and wind for a city.
        Input should be a city name, optionally followed by a country.
        """
        if not api_key:
            return (
                "Weather lookup is unavailable because WEATHERSTACK_API_KEY "
                "has not been configured."
            )

        try:
            response = requests.get(
                "https://api.weatherstack.com/current",
                params={
                    "access_key": api_key,
                    "query": city,
                    "units": "m",
                },
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()

            # Weatherstack may return an API error even when HTTP status is 200.
            if data.get("success") is False or "error" in data:
                error = data.get("error") or {}
                code = error.get("code", "unknown")
                info = error.get("info", "Unknown Weatherstack API error")
                return f"Weatherstack API error ({code}): {info}"

            current = data.get("current")
            location = data.get("location") or {}

            if not current:
                return f"Weatherstack returned no current weather data for {city}."

            descriptions = current.get("weather_descriptions") or []
            description = descriptions[0] if descriptions else "Not available"

            place = location.get("name") or city
            country = location.get("country") or ""
            place_label = f"{place}, {country}" if country else place

            return (
                f"Location: {place_label}\n"
                f"Temperature: {current.get('temperature', 'N/A')}°C\n"
                f"Weather: {description}\n"
                f"Humidity: {current.get('humidity', 'N/A')}%\n"
                f"Wind speed: {current.get('wind_speed', 'N/A')} km/h\n"
                f"Wind direction: {current.get('wind_dir', 'N/A')}"
            )

        except requests.Timeout:
            return f"The Weatherstack request timed out while checking {city}."
        except requests.RequestException as exc:
            # Avoid printing the full request URL because it may contain the API key.
            return (
                f"The Weatherstack request failed ({type(exc).__name__}). "
                "Please check your connection and API access."
            )
        except (ValueError, TypeError, KeyError):
            return f"Could not interpret the Weatherstack response for {city}."

    return get_weather_data


# ============================================================
# REACT PROMPT
# ============================================================

def get_react_prompt():
    """
    Use the same public LangSmith ReAct prompt as the supplied notebook.
    If it cannot be retrieved, use a local prompt with the same variables.
    """
    if Client is not None:
        try:
            langsmith_client = Client()
            return langsmith_client.pull_prompt(
                "hwchase17/react",
                dangerously_pull_public_prompt=True,
            )
        except Exception:
            pass

    return PromptTemplate.from_template(
        """Answer the following question as accurately as possible.
You have access to the following tools:

{tools}

Use this format:
Question: the user's question
Thought: what information is needed
Action: one of [{tool_names}]
Action Input: the input for the selected tool
Observation: the tool result
... (repeat Thought/Action/Action Input/Observation when needed)
Thought: I now know the final answer
Final Answer: the answer to the original question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    )


# ============================================================
# BUILD THE LANGCHAIN AGENT
# ============================================================

@st.cache_resource(show_spinner=False)
def build_agent_executor(
    openai_key: str,
    tavily_key: str,
    weatherstack_key: str,
    model_name: str,
) -> AgentExecutor:
    """Build and cache the LLM, tools and ReAct agent."""

    # TavilySearch reads its key from the environment.
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["TAVILY_API_KEY"] = tavily_key

    # Match the working notebook's ChatOpenAI Responses API configuration.
    llm = ChatOpenAI(
        model=model_name,
        api_key=openai_key,
        use_responses_api=True,
        temperature=0,
    )

    # Same Tavily integration and max_results setting as the notebook.
    search_tool = TavilySearch(max_results=2)
    tools: list[Any] = [search_tool]

    # Weatherstack is optional: search remains available if its key is absent.
    if weatherstack_key:
        tools.append(create_weather_tool(weatherstack_key))

    prompt = get_react_prompt()

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
        return_intermediate_steps=True,
    )


# ============================================================
# SIDEBAR: CONFIGURATION STATUS
# ============================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    st.caption("API keys are read from your local .env file.")

    if ENV_PATH:
        st.write("Environment file:", str(ENV_PATH))
    else:
        st.warning("No .env file was found. Checking system environment variables.")

    st.write("OpenAI API key:", "✅ Loaded" if OPENAI_API_KEY else "❌ Not found")
    st.write("Tavily API key:", "✅ Loaded" if TAVILY_API_KEY else "❌ Not found")
    st.write(
        "Weatherstack API key:",
        "✅ Loaded" if WEATHERSTACK_API_KEY else "⚠️ Not found — weather disabled",
    )
    st.write("OpenAI model:", f"`{OPENAI_MODEL}`")

    st.divider()
    st.caption("Your API keys are never displayed in this app.")


# ============================================================
# USER INPUT
# ============================================================

st.subheader("What would you like the agent to do?")

with st.form("agent_query_form", clear_on_submit=False):
    user_query = st.text_area(
        "Enter your query:",
        placeholder=(
            "Example: Find the capital of India, then check its current "
            "temperature, weather description and humidity."
        ),
        height=100,
        help="You can ask for web research, current weather, or a combination.",
    )
    run_agent = st.form_submit_button("🚀 Run Agent", type="primary")

if not WEATHERSTACK_API_KEY:
    st.info(
        "Web search is available, but current-weather lookups are disabled "
        "until WEATHERSTACK_API_KEY is added to your .env file."
    )


# ============================================================
# RUN THE AGENT
# ============================================================

if run_agent:
    if not user_query.strip():
        st.warning("Please enter a question before running the agent.")

    else:
        missing_keys = []
        if not OPENAI_API_KEY:
            missing_keys.append("OPENAI_API_KEY")
        if not TAVILY_API_KEY:
            missing_keys.append("TAVILY_API_KEY")

        if missing_keys:
            st.error("Missing required API key(s): " + ", ".join(missing_keys))
            expected_path = (
                ENV_PATH
                if ENV_PATH
                else SCRIPT_DIR / "research" / ".env"
            )
            st.info(f"Please check your .env file: {expected_path}")

        else:
            try:
                with st.spinner(
                    "Agent is thinking... Searching the web and gathering information."
                ):
                    executor = build_agent_executor(
                        OPENAI_API_KEY,
                        TAVILY_API_KEY,
                        WEATHERSTACK_API_KEY,
                        OPENAI_MODEL,
                    )
                    result = executor.invoke({"input": user_query.strip()})

                st.session_state["last_question"] = user_query.strip()
                st.session_state["last_answer"] = result.get(
                    "output",
                    "The agent did not return an answer.",
                )
                st.session_state["last_steps"] = result.get(
                    "intermediate_steps",
                    [],
                )

            except Exception as exc:
                st.error("The agent encountered an error.")
                st.caption(
                    "Check that your API keys, model access and installed "
                    "LangChain packages are configured correctly."
                )
                st.code(f"{type(exc).__name__}: {exc}")


# ============================================================
# DISPLAY THE RESULT
# ============================================================

if st.session_state.get("last_answer"):
    st.divider()
    st.subheader("Final Response")
    st.markdown(st.session_state["last_answer"])

    steps = st.session_state.get("last_steps", [])
    if steps:
        with st.expander("🔎 View agent activity"):
            for index, (action, observation) in enumerate(steps, start=1):
                st.markdown(f"**Step {index}: `{action.tool}`**")
                st.caption(f"Tool input: {action.tool_input}")
                st.write(observation)
                if index < len(steps):
                    st.divider()

    with st.expander("Show your question"):
        st.write(st.session_state.get("last_question", ""))

    if st.button("Clear result"):
        for key in ("last_question", "last_answer", "last_steps"):
            st.session_state.pop(key, None)
        st.rerun()

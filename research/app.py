"""
Agentic AI Assistant
Streamlit + LangChain create_agent + Tavily web search + Weatherstack current weather.

Expected .env variables:
    OPENAI_API_KEY=...
    TAVILY_API_KEY=...
    WEATHERSTACK_API_KEY=...
    APP_PASSWORD=...  # password required to open the Streamlit app
    OPENAI_MODEL=gpt-6-luna  # optional; defaults to the model used in the project

API keys are loaded from .env and are never displayed in the interface.
"""

from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
from typing import Any

import certifi
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


# =============================================================================
# ENVIRONMENT AND API KEYS
# =============================================================================

os.environ["SSL_CERT_FILE"] = certifi.where()

SCRIPT_DIR = Path(__file__).resolve().parent


def load_project_env() -> Path | None:
    """Find .env whether app.py is saved in the project root or research folder."""
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
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

# Keep the model setting used by the existing project.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna").strip() or "gpt-6-luna"


# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered",
)


# =============================================================================
# PASSWORD GATE
# =============================================================================

def require_app_password() -> None:
    """Require the shared application password before showing the agent UI."""

    # Fail closed: never expose the agent if the password was not configured.
    if not APP_PASSWORD:
        st.title("🔒 Private Agent Access")
        st.error("The application password has not been configured.")
        st.info(
            "Set APP_PASSWORD in your local .env file and in Render's "
            "Environment Variables, then restart or redeploy the app."
        )
        st.stop()

    if st.session_state.get("authenticated", False):
        return

    st.title("🔒 Private Agent Access")
    st.caption("This is a private application. Enter your password to continue.")

    with st.form("private_login_form", clear_on_submit=True):
        entered_password = st.text_input(
            "Application password",
            type="password",
            placeholder="Enter your password",
        )
        login_submitted = st.form_submit_button(
            "Log in",
            type="primary",
            use_container_width=True,
        )

    if login_submitted:
        entered_bytes = entered_password.encode("utf-8")
        expected_bytes = APP_PASSWORD.encode("utf-8")

        if hmac.compare_digest(entered_bytes, expected_bytes):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password. Please try again.")

    # Stop execution so unauthenticated visitors cannot reach the agent UI
    # or trigger any OpenAI, Tavily, or Weatherstack requests.
    st.stop()


require_app_password()

st.title("🤖 Agentic AI Assistant")
st.markdown(
    "Search + Weather AI Agent using **LangChain**, **Tavily** and **Weatherstack**."
)
st.caption(
    "Ask a question that needs web research, current weather information, or both."
)
st.divider()


# =============================================================================
# WEATHERSTACK TOOL
# =============================================================================

def create_weather_tool(api_key: str):
    """Create a LangChain tool that retrieves current weather from Weatherstack."""

    from langchain_core.tools import tool

    @tool("get_weather_data")
    def get_weather_data(city: str) -> str:
        """
        Fetch current weather for a city using Weatherstack.

        Use this tool whenever the user asks for current or live weather.
        Input should be a city name, optionally followed by a country.
        Returns temperature in Celsius, weather description, humidity,
        wind speed, wind direction and the local observation time.
        """
        if not api_key:
            return "Weather lookup is unavailable because WEATHERSTACK_API_KEY is not configured."

        try:
            response = requests.get(
                "https://api.weatherstack.com/current",
                params={
                    "access_key": api_key,
                    "query": city,
                    "units": "m",
                },
                timeout=20,
                verify=certifi.where(),
            )
            response.raise_for_status()
            data = response.json()

            # Weatherstack can return an API error even when HTTP status is 200.
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
            local_time = location.get("localtime") or "Not available"

            return (
                f"Location: {place_label}\n"
                f"Local time: {local_time}\n"
                f"Temperature: {current.get('temperature', 'N/A')}°C\n"
                f"Weather: {description}\n"
                f"Humidity: {current.get('humidity', 'N/A')}%\n"
                f"Wind speed: {current.get('wind_speed', 'N/A')} km/h\n"
                f"Wind direction: {current.get('wind_dir', 'N/A')}"
            )

        except requests.Timeout:
            return f"The Weatherstack request timed out while checking {city}."
        except requests.RequestException:
            # Do not expose the request URL because it may contain the API key.
            return (
                "The Weatherstack request failed. "
                "Please check the network connection and Weatherstack API access."
            )
        except (ValueError, TypeError, KeyError):
            return f"Could not interpret the Weatherstack response for {city}."

    return get_weather_data


# =============================================================================
# AGENT INSTRUCTIONS
# =============================================================================

SYSTEM_PROMPT = """
You are an accurate research and weather assistant.

AVAILABLE TOOLS
- tavily_search: searches the web and returns search results.
- get_weather_data: retrieves current weather from Weatherstack.

TOOL SELECTION
1. If the user asks for current/live weather, call get_weather_data.
2. If the user asks for web research, facts, opening hours, or source URLs,
   call tavily_search.
3. If the user asks for both weather and web research, use both tools.
   Gather the required information before writing the final answer.
4. Do not invent current weather, search results, facts, or URLs.

OFFICIAL-SOURCE AND SOURCE-QUALITY RULES
1. If the user explicitly asks for an official source, official website,
   official opening hours, or information from a named organisation, use
   Tavily's include_domains argument to restrict the search to the relevant
   official domain whenever it can be identified.
2. If the official domain is not known, first identify the organisation's
   official website. Then search that domain for the requested information.
3. Do not describe a third-party website as an official source.
4. If an official source cannot be found, clearly say that it could not be
   verified from an official source. Do not silently substitute a third-party
   source.
5. Prefer primary sources and official organisation websites over blogs,
   travel sites, aggregators, and general reference sites.
6. If sources disagree, explain the discrepancy and prefer the official
   source. Do not combine conflicting details as though they agree.

ANSWER FORMAT
- Answer the user's question directly.
- Use clear headings and concise bullet points when helpful.
- For web-derived facts, include clickable Markdown links using the exact
  URLs returned by Tavily. Never fabricate or rewrite a source URL.
- Make clear which details come from Weatherstack and include its local
  observation time when available.
- If a requested detail is missing from the tool results, say it is unavailable
  rather than guessing.
- For recommendations, explain the reasoning and distinguish facts from
  recommendations.
"""


# =============================================================================
# BUILD AND CACHE THE LANGCHAIN AGENT
# =============================================================================

@st.cache_resource(show_spinner=False)
def build_agent(
    openai_key: str,
    tavily_key: str,
    weatherstack_key: str,
    model_name: str,
):
    """Build and cache the LLM, tools and LangChain agent."""

    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["TAVILY_API_KEY"] = tavily_key

    llm = ChatOpenAI(
        model=model_name,
        api_key=openai_key,
        use_responses_api=True,
    )

    # Tavily supports include_domains at tool-call time. The system prompt
    # instructs the agent to use it for requests requiring official sources.
    search_tool = TavilySearch(
        max_results=5,
        topic="general",
        search_depth="basic",
        include_answer=False,
        include_raw_content="markdown",
    )
    search_tool.description = (
        "Search the web for reliable information. "
        "Use include_domains to restrict results to an official website when "
        "the user requests official information or names an organisation. "
        "The tool accepts query and optional include_domains/exclude_domains."
    )

    tools = [search_tool]

    # Weatherstack is optional; Tavily web search remains available without it.
    if weatherstack_key:
        tools.append(create_weather_tool(weatherstack_key))

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


# =============================================================================
# HELPERS: FORMAT AGENT OUTPUT AND TOOL ACTIVITY
# =============================================================================

def content_to_text(content: Any) -> str:
    """Convert LangChain message content into readable text."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block["text"]))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return "\n".join(part for part in parts if part).strip()

    return str(content)


def summarize_tavily_result(content: Any) -> str:
    """Turn Tavily's full JSON response into a short readable activity summary."""
    raw_text = content_to_text(content)

    try:
        data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
    except (json.JSONDecodeError, TypeError):
        return raw_text[:2500]

    if not isinstance(data, dict):
        return str(data)[:2500]

    lines: list[str] = []

    query = data.get("query")
    if query:
        lines.append(f"**Search query:** {query}")

    results = data.get("results") or []
    if not results:
        lines.append("No search results were returned.")
        return "\n\n".join(lines)

    lines.append(f"**Results returned:** {len(results)}")

    for index, item in enumerate(results[:5], start=1):
        if not isinstance(item, dict):
            continue

        title = item.get("title") or "Untitled result"
        url = item.get("url") or ""
        snippet = item.get("content") or ""
        snippet = " ".join(str(snippet).split())

        if len(snippet) > 450:
            snippet = snippet[:450].rstrip() + "…"

        lines.append(f"**{index}. {title}**")
        if url:
            lines.append(f"Source: {url}")
        if snippet:
            lines.append(str(snippet))

    return "\n\n".join(lines)


def summarize_tool_result(tool_name: str, content: Any) -> str:
    """Create a concise observation for the Streamlit activity log."""
    if tool_name == "tavily_search":
        return summarize_tavily_result(content)

    text = content_to_text(content)
    if len(text) > 2500:
        return text[:2500].rstrip() + "…"
    return text


def extract_agent_output(result: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Extract the final AI response and concise tool activity from create_agent output."""
    messages = result.get("messages", []) if isinstance(result, dict) else []

    # Match tool-call IDs to their original names and arguments.
    tool_calls_by_id: dict[str, dict[str, Any]] = {}
    for message in messages:
        for tool_call in getattr(message, "tool_calls", []) or []:
            call_id = tool_call.get("id")
            if call_id:
                tool_calls_by_id[call_id] = tool_call

    steps: list[dict[str, Any]] = []

    for message in messages:
        if getattr(message, "type", "") != "tool":
            continue

        call_id = getattr(message, "tool_call_id", None)
        tool_call = tool_calls_by_id.get(call_id, {})

        tool_name = (
            getattr(message, "name", None)
            or tool_call.get("name")
            or "tool"
        )
        tool_input = tool_call.get("args") or {}
        observation = summarize_tool_result(
            tool_name,
            getattr(message, "content", ""),
        )

        steps.append(
            {
                "tool": tool_name,
                "tool_input": tool_input,
                "observation": observation,
            }
        )

    # The final answer is the last AI message that is not another tool call.
    final_answer = ""
    for message in reversed(messages):
        if getattr(message, "type", "") != "ai":
            continue
        if getattr(message, "tool_calls", None):
            continue

        candidate = content_to_text(getattr(message, "content", ""))
        if candidate.strip():
            final_answer = candidate.strip()
            break

    if not final_answer:
        final_answer = "The agent did not return a final answer."

    return final_answer, steps


# =============================================================================
# SIDEBAR: CONFIGURATION STATUS
# =============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    st.caption("API keys and the app password are read from environment variables. Locally, these can be stored in .env.")

    if ENV_PATH:
        st.write("Environment file:", str(ENV_PATH))
    else:
        st.warning(
            "No .env file was found. Checking system environment variables."
        )

    st.write(
        "OpenAI API key:",
        "✅ Loaded" if OPENAI_API_KEY else "❌ Not found",
    )
    st.write(
        "Tavily API key:",
        "✅ Loaded" if TAVILY_API_KEY else "❌ Not found",
    )
    st.write(
        "Weatherstack API key:",
        "✅ Loaded" if WEATHERSTACK_API_KEY
        else "⚠️ Not found — weather disabled",
    )
    st.write("OpenAI model:", f"`{OPENAI_MODEL}`")

    st.divider()
    st.caption("Your API keys are never displayed in this app.")

    if st.button("Log out", use_container_width=True):
        st.session_state["authenticated"] = False
        for key in ("last_question", "last_answer", "last_steps"):
            st.session_state.pop(key, None)
        st.rerun()


# =============================================================================
# USER INPUT
# =============================================================================

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


# =============================================================================
# RUN THE AGENT
# =============================================================================

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
                    agent = build_agent(
                        OPENAI_API_KEY,
                        TAVILY_API_KEY,
                        WEATHERSTACK_API_KEY,
                        OPENAI_MODEL,
                    )

                    result = agent.invoke(
                        {
                            "messages": [
                                {
                                    "role": "user",
                                    "content": user_query.strip(),
                                }
                            ]
                        },
                        config={"recursion_limit": 12},
                    )

                final_answer, steps = extract_agent_output(result)

                st.session_state["last_question"] = user_query.strip()
                st.session_state["last_answer"] = final_answer
                st.session_state["last_steps"] = steps

            except Exception as exc:
                st.error("The agent encountered an error.")
                st.caption(
                    "Check your API keys, model access and installed "
                    "LangChain packages."
                )

                # Avoid accidentally displaying an API key in an error message.
                safe_error = str(exc)
                for secret in (
                    OPENAI_API_KEY,
                    TAVILY_API_KEY,
                    WEATHERSTACK_API_KEY,
                ):
                    if secret:
                        safe_error = safe_error.replace(secret, "[REDACTED]")

                st.code(f"{type(exc).__name__}: {safe_error}")


# =============================================================================
# DISPLAY THE RESULT
# =============================================================================

if st.session_state.get("last_answer"):
    st.divider()
    st.subheader("Final Response")
    st.markdown(st.session_state["last_answer"])

    steps = st.session_state.get("last_steps", [])

    with st.expander("🔎 View agent activity"):
        if not steps:
            st.caption("The agent answered without calling an external tool.")
        else:
            for index, step in enumerate(steps, start=1):
                st.markdown(f"**Step {index}: `{step['tool']}`**")

                st.caption("Tool input:")
                if isinstance(step["tool_input"], (dict, list)):
                    st.json(step["tool_input"])
                else:
                    st.write(step["tool_input"])

                st.markdown("**Tool result summary:**")
                st.markdown(step["observation"])

                if index < len(steps):
                    st.divider()

    with st.expander("Show your question"):
        st.write(st.session_state.get("last_question", ""))

    if st.button("Clear result"):
        for key in ("last_question", "last_answer", "last_steps"):
            st.session_state.pop(key, None)
        st.rerun()

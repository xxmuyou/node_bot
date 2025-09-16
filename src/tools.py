
from typing import Text, Any, cast
from datetime import datetime, timezone
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.runtime import get_runtime

from context import Context
from utils import _load_api_key

def get_tools():
    return [tavily_search, get_current_time]

async def tavily_search(query: Text) -> Text:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    runtime = get_runtime(Context)
    tavily_search = load_tavily_search(max_search_results=runtime.context.max_search_results)
    return cast(dict[str, Any], await tavily_search.ainvoke({"query": query}))

async def get_current_time(region: Text) -> Text:
    """Get the current time in the specified region"""
    region = region.lower()
    return datetime.now(timezone(region)).isoformat()

def load_tavily_search(max_search_results: int = 10) -> TavilySearch:
    api_key = _load_api_key("TAVILY_API_KEY")
    return TavilySearch(api_key=api_key, max_results=max_search_results)

if __name__ == "__main__":
    tavily_search = load_tavily_search()
    print(tavily_search.invoke("What is the weather in Tokyo?"))


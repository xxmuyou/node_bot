import os
from dataclasses import dataclass
from typing import Text
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from openai.types.responses import response
from langgraph.graph import StateGraph

from context import AgnetConfig

load_dotenv()

def _load_api_key(key_name: Text) -> Text:
    """
    Load the api key from the .env file, Api key_name include: {DEEPSEEK_API_KEY, TAVILY_API_KEY} or other tool api keys
    """
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"Not found related {key_name} api keys, please set in .env file")
    return api_key


def load_chat_model(config: AgnetConfig) -> ChatOpenAI:
    config = config or AgnetConfig()
    api_key = _load_api_key(config.key_name)
    return ChatOpenAI(base_url=config.base_url, api_key=api_key, model=config.model, temperature=config.temperature)

def plot_graph(graph: StateGraph, save_path: Text = "graph.png") -> None:
    graph.get_graph().draw_mermaid_png(output_file_path=save_path)

if __name__ == "__main__":
    config = AgnetConfig()
    llm = load_chat_model(config)
    response = llm.invoke("Hello, how are you?")
    print(response)
    tavily_api_key = _load_api_key("TAVILY_API_KEY")


from typing import Text, Annotated
from dataclasses import dataclass, field

from prompts import Prompts

@dataclass
class AgnetConfig:
    key_name: Text = "DEEPSEEK_API_KEY"
    base_url: Text = "https://api.deepseek.com"
    model: Text = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = None

@dataclass
class Context:
    """ The Context class is used to initialize global variables for configuration. """
    model_config: Annotated[
        AgnetConfig,
        "AI model configuration object, containing settings such as base URL, model name, temperature parameter, and other configurations."
    ] = field(
        default_factory=AgnetConfig,
        metadata={"description": "Config should be contain model's name, base_url, key_name for OpenAI"}
    )

    system_prompt: str = field(
        default=Prompts.system_prompt,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )
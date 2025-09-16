
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

def set_short_memory() -> BaseCheckpointSaver:
    return InMemorySaver()
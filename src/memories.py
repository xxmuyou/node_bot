from typing import Union

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver, PersistentDict
from langchain_core.runnables import RunnableConfig
from collections import defaultdict


def set_memory(local_storage: bool = False) -> BaseCheckpointSaver:
    """
    Create memory checkpointer based on storage preference
    
    Args:
        local_storage: If True, use long-term database storage; 
                      If False, use short-term in-memory storage
        
    Returns:
        BaseCheckpointSaver object compatible with LangGraph
    """
    if local_storage:
        raise NotImplementedError("Long-term memory is not implemented Currently")
        return long_memory()
    else:
        return short_memory()

def short_memory() -> BaseCheckpointSaver:
    """
    Create short-term memory checkpointer (to be implemented)
    
    Returns:
        Short-term memory checkpointer
    """
    # TODO : Set short-term memory logic
    return InMemorySaver()

def long_memory() -> BaseCheckpointSaver:
    """
    Create long-term memory checkpointer (to be implemented)
    
    Returns:
        Long-term memory checkpointer with database storage
    """
    # TODO : Set long-term memory logic, like SQLite ,and class inheritance: BaseCheckpointSaver
    pass
    

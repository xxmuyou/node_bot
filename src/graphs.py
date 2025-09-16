from typing import Dict, List, cast, Literal
from datetime import datetime

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointSaver

from context import Context
from state import State
from utils import load_chat_model, plot_graph
from tools import get_tools
from memories import set_memory


async def call_model(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """Call the LLM powering our "agent".

    This function prepares the prompt, initializes the model, and processes the response.

    Args:
        state (State): The current state of the conversation.
        config (RunnableConfig): Configuration for the model run.

    Returns:
        dict: A dictionary containing the model's response message.
    """
    # Initialize the model with tool binding. Change the model or add more tools here.
    model = load_chat_model(runtime.context.model_config).bind_tools(get_tools())

    # Format the system prompt. Customize this to change the agent's behavior.
    system_message = runtime.context.system_prompt.format(
        system_time=datetime.now().isoformat()
    )

    # Get the model's response
    response = cast(
        AIMessage,
        await model.ainvoke(
            [{"role": "system", "content": system_message}, *state.messages]
        ),
    )

    # Handle the case when it's the last step and the model still wants to use a tool
    if state.is_last_step and response.tool_calls:
        return {
            "messages": [
                AIMessage(
                    id=response.id,
                    content="Sorry, I could not find an answer to your question in the specified number of steps.",
                )
            ]
        }

    # Return the model's response as a list to be added to existing messages
    return {"messages": [response]}

def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """Determine the next node based on the model's output.

    This function checks if the model's last message contains tool calls.

    Args:
        state (State): The current state of the conversation.

    Returns:
        str: The name of the next node to call ("__end__" or "tools").
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )
    # If there is no tool call, then we finish
    if not last_message.tool_calls:
        return "__end__"
    # Otherwise we execute the requested actions
    return "tools"


def build_graph(checkpointer: BaseCheckpointSaver):
    # Define a new graph

    builder = StateGraph(State)

    # Define the two nodes we will cycle between
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(get_tools()))

    # Set the entrypoint as `call_model`
    # This means that this node is the first one called
    builder.add_edge("__start__", "call_model")

    # Add a conditional edge to determine the next step after `call_model`
    builder.add_conditional_edges(
        "call_model",
        # After call_model finishes running, the next node(s) are scheduled
        # based on the output from route_model_output
        route_model_output,
    )

    # Add a normal edge from `tools` to `call_model`
    # This creates a cycle: after using tools, we always return to the model
    builder.add_edge("tools", "call_model")

    # Compile the builder into an executable graph
    graph = builder.compile(name="ReAct Agent", checkpointer=checkpointer)

    return graph


from langgraph.checkpoint.memory import InMemorySaver
from typing import AsyncIterator, Any
async def print_stream():
    memory = set_memory(local_storage=False)
    config = {
        "configurable": {"thread_id": "1"}
    }
    input_msg = {"messages": [HumanMessage(content="现在是具体什么时间")]}
    context = Context()
    graph: StateGraph = build_graph(checkpointer=memory)
    aresponse: AsyncIterator[dict[str, Any] | Any] = graph.astream(input_msg, config=config, context=context, stream_mode="values")
    pass

if __name__ == "__main__":
    import asyncio
    memory = set_memory(local_storage=False)
    config = {
        "configurable": {"thread_id": "1"}
    }
    context = Context()
    graph: StateGraph = build_graph(checkpointer=memory)
    # input_msg = {"messages": [HumanMessage(content="现在是具体什么时间")]}
    while True:
        input_msg = {"messages": [HumanMessage(content=input("请输入问题: "))]}
        # plot_graph(graph, save_path="graph.png")
        if input_msg["messages"][0].content in ["bye", "exit", "quit"]:
            break

        response = asyncio.run(graph.ainvoke(input_msg, config=config, context=context))
        print(response.get("messages", [])[-1].content)

    

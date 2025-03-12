import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

openai_api_key = os.environ['OPENAI_API_KEY']
tavily_api_key = os.environ['TAVILY_API_KEY']

from langgraph.prebuilt import create_react_agent
from langchain_community.llms import OpenAI 
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

#* set up the tavily client
tavily_client = TavilyClient(api_key=tavily_api_key)

#* set up the tools
@tool
def research(query: str) -> str:
    """Useful for when you need to answer questions about the Catholic faith. 
    You should use sources from the Vatican if you can.
    Output should be in the form of a string."""

    answer = tavily_client.qna_search(query=query)

    return f"Answer: {answer}"

tools = [research]

#* define the state
from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """The state of the agent."""

    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]


#* initiate the model we are going to use and bind the tools
model = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_api_key)

model = model.bind_tools(tools)

agent = create_react_agent(model, tools=tools)

import json
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

tools_by_name = {tool.name: tool for tool in tools}


# Define our tool node
def tool_node(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}




# Define the node that calls the model
def call_model(
    state: AgentState,
    config: RunnableConfig,
):
    # this is similar to customizing the create_react_agent with 'prompt' parameter, but is more flexible
    system_prompt = SystemMessage(
        "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
    )
    response = model.invoke([system_prompt] + state["messages"], config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


from langgraph.graph import StateGraph, END

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "tools",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", "agent")

# Now we can compile and visualize our graph
graph = workflow.compile()

from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

 # Helper function for formatting the stream nicely
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

from langchain_core.messages import AIMessage, HumanMessage

def main():
    messages = []

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Exiting. Goodbye!")
            break
        human_msg = HumanMessage(user_input)
        messages.append(human_msg)

        response = agent.invoke({"messages": messages})
        # print(f"RESPONSE: {response}")
        messages.append(response)
        print(f"RESPONSE: {response}")
        #Want to summarize at this point before replying
        #summarize_conversation({"messages": messages})
        """for m in response['messages']:
            print(f"m: {m}")"""

        
        #print(f"\n\nAIMessage with ToolCalls: {response.tool_calls}")
        if 'tool_calls' in response.additional_kwargs:
            for tool_call in response.tool_calls:
                selected_tool = {'research': research}[tool_call["name"].lower()]
                tool_msg = selected_tool.invoke(tool_call)
                messages.append(tool_msg)
            query_answer = agent.invoke(messages)
            print(f"Agent: {query_answer.content}")
        else: 
            print(f"Agent: {response.content}")


if __name__ == "__main__":
    main()
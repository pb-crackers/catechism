import re
import json
from uuid import uuid4
from typing import Optional, List

from agenttools import search_catechism

from langchain.tools import Tool, tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from pydantic import BaseModel

# Import the ConversationMemory class from your database module.
# from database import ConversationMemory



# Remove local in-memory storage; we now use DynamoDB.
# conversations = {}

#############################################
#            TOOLS DECLARATION              #
#############################################

#############################################
#   LIST OF TOOLS TO BIND TO THE LLM         #
#############################################
# Now include both flip_calc and remember tools.
tools = [search_catechism]

#############################################
#       TOOL ERROR HANDLING / FALLBACK      #
#############################################
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    from langgraph.prebuilt import ToolNode
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

#############################################
#      INITIALIZE THE LANGUAGE MODEL        #
#############################################
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o", temperature=1)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=True)

#############################################
#        SET UP ASSISTANT & GRAPH           #
#############################################
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage, HumanMessage

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: dict, config: RunnableConfig):
        result = self.runnable.invoke(state)
        return {"messages": result}

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Cat, a Catholic, friendly assistant designed to help users find answers from the Catechism of the Catholic Church."
            "You are capable of querying a vector database using the search_catechism tool to find the most relevant information for their query."
            "You do NOT edit the information that you get back from those searches, however you are allowed to use simpler language while maintaining the same semantic meaning of the language."
            "You do not assist users with anything other than querying the Catechism."
            "You only reply in English."
            "You can answer basic questions about the Catechsim content, but should direct the user to speak to their local priest if they want to learn more."
        ),
        ("placeholder", "{messages}")
    ]
)

assistant_runnable = primary_assistant_prompt | llm_with_tools

# Define the conversation graph.
builder = StateGraph(MessagesState)
builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("action", create_tool_node_with_fallback(tools))
builder.set_entry_point("assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
    {"action": "action", END: END}
)
builder.add_edge("action", "assistant")
graph = builder.compile()

#############################################
#      SET UP DATABASE (DynamoDB) STORAGE   #
#############################################
# Create a global ConversationMemory instance.
# (Ensure your environment variables or parameters set the table name and region correctly.)
#db_memory = ConversationMemory() #ddatabase.ConversationMemory()

#############################################
#          HELPER: GENERATE SUMMARY          #
#############################################
def generate_summary(existing_summary: str, user_input: str, assistant_response: str) -> str:
    """
    Generates a distilled summary of the conversation based on the previous summary,
    the latest user input, and the assistant's response. This summary is designed
    to keep token usage down while preserving key details.
    """
    summary_prompt = (
        "Please summarize the following conversation by extracting only the key information, "
        "especially any important details such as tool call results and critical facts. "
        "Do not include every detail—only what is necessary for context in future messages.\n\n"
        f"Existing Summary: {existing_summary}\n"
        f"User: {user_input}\n"
        f"Assistant: {assistant_response}\n\n"
        "New Summary:"
    )
    from langchain_core.messages import HumanMessage
    # Change here: pass a list of messages instead of a dict.
    summary_response = llm.invoke([HumanMessage(summary_prompt)])
    return summary_response.content.strip()


def main():
    print("Interactive mode. Type 'quit', 'exit', or 'q' to stop.")
    while True:
        """db_memory = ConversationMemory() #ddatabase.DevConversationMemory()
        conversation_id = "0002"
        conversation = db_memory.get_conversation(conversation_id)
        if conversation is None:
        # No existing conversation: create a new record with both fields empty.
            db_memory.create_conversation(conversation_id, summary="", last_user_input="")
            conversation = {"summary": "", "last_user_input": "", "detailed_history": ""}"""

        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Exiting. Goodbye!")
            break
        messages = []
        """if conversation.get("summary"):
            messages.append(
                AIMessage(content=f"Conversation summary so far: {conversation['summary']}")
            )"""
        human_msg = HumanMessage(user_input)
        messages.append(human_msg)

        response = assistant_runnable.invoke({"messages": messages})
        #print(f"\n\nRESPONSE: {response}\n\n")
        messages.append(response)

        # example message for testing: hi there can you get me the flip profit and net brrr on a deal with a price of 70k, a arv of 160k, and a rehab budget of 40k


        while response.additional_kwargs.get("tool_calls"):
            for tool_call in response.tool_calls:
                print(f"\n\nDEBUG TOOL CALL: {tool_call}\n\n")
                try:
                    tool_name = tool_call["name"].lower()
                    if tool_name in ["search_catechsim"]:
                        selected_tool = {"search_catechsim": search_catechism}[tool_name]
                    else:
                        continue
                    print(f"\n\nSelected tool: {tool_name}\n\n")
                    tool_msg = selected_tool.invoke(tool_call)
                    print(f"\n\nTool message: {tool_msg}\n\n")
                    messages.append(tool_msg)
                except Exception as tool_error:
                    return f"An error occurred while processing the tool: {tool_error}. Please contact support."
            #print(messages)
            response = llm_with_tools.invoke(messages)
        #print(f"RESPONSE: {response}")
        
        print(f"Agent: {response.content}")

        """previous_summary = conversation.get("summary", "")
        new_summary = generate_summary(previous_summary, user_input, response.content)
        
        # Update detailed_history: append the full exchange (consider truncating if too long).
        previous_history = conversation.get("detailed_history", "")
        new_history = previous_history + f"\nUser: {user_input}\nAssistant: {response.content}"
        # Optionally, you could add logic here to truncate new_history if it exceeds a certain length.
        
        # Update the conversation memory with the new summary and detailed history.
        db_memory.update_conversation(
            conversation_id,
            summary=new_summary,
            last_user_input=user_input,
            detailed_history=new_history
        )"""


# --- Run the CLI if this script is executed directly ---
if __name__ == "__main__":
    # You can run this for interactive testing.
    # To run the FastAPI server, use: uvicorn main:app --reload
    main()
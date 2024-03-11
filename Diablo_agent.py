import os
from uuid import uuid4

# The base setting of api
unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f" Corrector_agent_test_38"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__eb24c4743a814171b07dc0591921a299"  # Update to your API key
os.environ["TAVILY_API_KEY"] = "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"
# Used by the agent in this tutorial
os.environ["OPENAI_API_KEY"] = "sk-OqlsJwGTMdD1ujTG02Bb0fE08b7f4b30B07d8e83012bA8A8"
os.environ["OPENAI_API_BASE"] = "https://oneapi.xty.app/v1"

from langsmith import Client

client = Client()

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, BaseMessage, FunctionMessage, HumanMessage
import operator
from typing import Annotated, List, Sequence, Tuple, TypedDict, Union

from langchain.agents import create_openai_functions_agent
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from tools.robot_func import python_repl

tools = [python_repl]
# corrector_prompt
corrector_prompt = hub.pull("blackj/corrector_prompt")
print(corrector_prompt)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# corrector_prompt
# tools = []
# print(corrector_prompt.messages)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    corrector_state: bool


corrector_template = \
    """You job is to make the user_requirenment clear.

Just Output the Result,dont output decription.Dont make judge but help. 

You should work as the procedure including in the ###

### 1. Interpret the Command of user: Analyze the user's input to determine if they want the robot to "Get" something or "Go" somewhere. Be aware that abbreviations might be used. 

Example: Interpret "GO" as "go to", and "MV" as "move to". ###

###

2. Correct and Clarify: Look for any spelling errors in the user's command that might indicate a specific location or item, it might something or somewhere in the house. Clarify the command without adding extra information. Note: Focus solely on the user's input for this step. Action: Correct any spelling mistakes and clarify the meaning of places or items mentioned.

Example: User Input: "Robt, GO to the kithcn and gt me a sppon." Corrected Command: "Robot,go to the kitchen and get me a spoon." ###
"""


def correct_user_input(state):
    last_message = state["messages"]
    print(last_message)
    # print(state["messages"])
    if state["corrector_state"]:
        return [SystemMessage(
            content=f"The user has confirm the command,I will send it to robot!!,the corrector command is {state['messages'][-2]}",
            kwarg={"corrector_state": True})] + last_message
    else:
        return [SystemMessage(content=corrector_template, kwarg={"corrector_state": False})] + last_message


def parse(ai_message: BaseMessage):
    """Parse the AI message."""
    return {"messages": [ai_message]}


# llm_with_tool = llm.bind_tools(tools)
from langchain_core.output_parsers import StrOutputParser

corrector_chain = correct_user_input | llm | parse

from tools.robot_func import robot_move, robot_catch, python_repl

tools = [robot_catch, robot_move, python_repl]


def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    functions = [format_tool_to_openai_function(t) for t in tools]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant, collaborating with other assistants."
                " Use the provided tools to progress towards answering the question."
                " If you are unable to fully answer, that's OK, another assistant with different tools "
                " will help where you left off. Execute what you can to make progress."
                " If you or any of the other assistants have the final answer or deliverable,"
                " prefix your response with FINAL ANSWER so the team knows to stop."
                " You have access to the following tools: {tool_names}.\n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_functions(functions)


def agent_node(state, agent, name):
    result = agent.invoke(state)
    # We convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, FunctionMessage):
        pass
    else:
        result = HumanMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "corrector_state": state["corrector_state"],
    }


robot_agent = create_agent(
    llm,
    [robot_catch, robot_move],
    system_message="You should provide accurate data for the chart generator to use.",
)

import functools

robot_node = functools.partial(agent_node, agent=robot_agent, name="Diablo")

# def _is_tool_call(msg):
#     return hasattr(msg, "additional_kwargs") and 'tool_calls' in msg.additional_kwargs


# runnable_agent = create_openai_tools_agent(llm, tools, corrector_prompt)
# agent_executor = AgentExecutor(
#     agent=runnable_agent, handle_parsing_errors=True
# )
#
# results = agent_executor.invoke(input={"user_requirenment": "robot,get the spoon"}, return_exceptions=True, verbose=True)
#
# print(results)

# write a binary search tree for me

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

memory = SqliteSaver.from_conn_string(":memory:")


def corrector_router(state):
    if state["corrector_state"]:
        print("Correction finished！！")
        return "Diablo"
    else:
        print("need to be confirm...")
        return END


nodes = {k: k for k in ["Diablo", END]}
workflow = StateGraph(AgentState)

workflow.add_node("corrector", corrector_chain)
workflow.add_node("Diablo", robot_node)
workflow.add_edge("Diablo", END)
workflow.add_conditional_edges("corrector", corrector_router, nodes)
workflow.set_entry_point("corrector")
graph = workflow.compile(checkpointer=memory)
import uuid

config = {"configurable": {"thread_id": str(uuid.uuid4())}}
corrector_state = False
while True:
    user = input('User (q/Q to quit): ')
    if user in {'q', 'Q'}:
        print('AI: Byebye')
        break
    if user in {'Y', 'y'}:
        user = "Confirm order!!Give it to Robot!!"
        corrector_state = True
    for output in graph.stream({"messages": [HumanMessage(content=user)], "corrector_state": corrector_state},
                               config=config):
        if "__end__" in output:
            continue
        # stream() yields dictionaries with output keyed by node name
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")

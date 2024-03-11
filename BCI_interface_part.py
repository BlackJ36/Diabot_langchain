os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"BCI_INTER"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__eb24c4743a814171b07dc0591921a299"  # Update to your API key
os.environ["TAVILY_API_KEY"] = "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"
# Used by the agent in this tutorial
os.environ["OPENAI_API_KEY"] = "sk-OqlsJwGTMdD1ujTG02Bb0fE08b7f4b30B07d8e83012bA8A8"
os.environ["OPENAI_API_BASE"] = "https://oneapi.xty.app/v1"

# from tools.robot_func import python_repl
# tools = [python_repl]

# from langgraph.prebuilt import ToolExecutor

# import ChatPromptTemplate
# 编程工具节点
# tool_executor = ToolExecutor(tools)

from langchain_openai import ChatOpenAI

# create a ChatOpenAI model with prompt
from langchain import hub
corrector_prompt = hub.pull("blackj/corrector_prompt")
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")




def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    from langchain_core.utils.function_calling import convert_to_openai_function

    functions = [convert_to_openai_function(t) for t in tools]
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


model = create_agent(llm, tools)

from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage


# Define the function that determines whether to continue or not
def should_continue(messages):
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(messages):
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return response


# Define the function to execute tools
def call_tool(messages):
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )
    # We call the tool_executor and get back a response
    response = tool_executor.invoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return function_message


from langgraph.graph import MessageGraph, END

# Define a new graph
workflow = MessageGraph()

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

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
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

from langchain_core.tools import tool
from typing import Annotated, List, Optional, Union
from langsmith import trace
from langchain_experimental.utilities import PythonREPL


@tool
def robot_move(position: Annotated[str, "The position of the robot."],
               direction: Annotated[str, "The direction to move the robot."]) -> Annotated[
    str, "The new position of the robot."]:
    """Move a robot in a given direction.

    Args:
        position: The position of the robot.
        direction: The direction to move the robot.

    Returns:
        The new position of the robot.
    """
    return f"the robot has reached the {direction} from {position}, what the robot should do next？"


@tool
def robot_catch(object_name: Annotated[str, "The name of the object."]) -> Annotated[str, "The name of the object."]:
    """Catch an object.

    Args:
        object_name: The name of the object.

    Returns:
        A message indicating the object has been caught.
    """
    return f"the robot has caught the {object_name}, what the robot should do next？"


repl = PythonREPL()


@tool
def python_repl(code: Annotated[str, "The code to execute."]) -> Annotated[str, "The result of the code."]:
    """Execute Python code.

    Args:
        code: The code to execute.

    Returns:
        The result of the code.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Faild to execute. Error:{repr(e)}"
    return f"Sussessfully executed:\n```Python\n{code}\n```\n Result:{result}"

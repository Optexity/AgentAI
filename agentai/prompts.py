from computergym.actions.action import (
    ActionTypes,
    ClickAction,
    InputText,
    ScrollAction,
    action_examples,
)
from pydantic import BaseModel, Field


class Response(BaseModel):
    """
    The response format for the action to take. Think step-by-step through the action you want to take.
    """

    reasoning: str = Field(description="Your reasoning for taking this action.")
    action_name: str = Field(
        description="The action_name should be one of the available actions"
    )
    action_params: dict = Field(
        description="""The parameters of the action you want to take. Must be valid JSON. 
        The action_params should be valid for that action.
        The action_params should be a dictionary with the parameters of the action.
            {
            "param1": "value1",
            "param2": "value2"
        }
        """
    )


click_example_response = Response(
    reasoning="I need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.",
    action_name=ActionTypes.click.value,
    action_params=action_examples[ActionTypes.click].model_dump(),
)

input_text_example_response = Response(
    reasoning="I need to enter text in text field with bid 12. I will use the input_text action.",
    action_name=ActionTypes.input_text.value,
    action_params=action_examples[ActionTypes.input_text].model_dump(),
)

instruction_prompt = """You are a UI Assistant, your goal is to help the user perform tasks using a web browser. You can
communicate with the user via a chat, to which the user gives you instructions and to which you
can send back messages. You have access to a web browser that both you and the user can see,
and with which only you can interact via specific commands.

Review the instructions from the user, the current state of the page and all other information
to find the best possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions."""


example_actions = f"""Here are examples of actions with chain-of-thought reasoning:
Click Example:
{click_example_response.model_dump_json(indent=4)}

Input Text Example:
{input_text_example_response.model_dump_json(indent=4)}
"""


next_action = """You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page and the task in hand before deciding on your next action."""

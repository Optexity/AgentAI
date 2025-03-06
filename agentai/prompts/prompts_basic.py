from computergym.actions.action import ActionTypes, action_examples

from .utils import Response

click_example_response = Response(
    observation="The current page looks like a homepage with signin button relevant for this task.",
    reasoning="I need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.",
    action_name=ActionTypes.click.value,
    action_params=action_examples[ActionTypes.click].model_dump(),
)

input_text_example_response = Response(
    observation="The current page looks signinpage with email and password fields relevant for this task.",
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
and executed by a program, make sure to follow the formatting instructions. Try not repeating same actions on same page as that would not be helpful to the user."""

format_instruction = """Only output the json object. Do not output anything other than the json object which should be directly parsable by pydantic. Output the json object in the following format:
```json
{
    "observation": "Summary of the observation you are responding to. This is the current state of the page.",
    "reasoning": "Your reasoning for taking this action.",
    "action_name": "The action_name should be one of the available actions",
    "action_params": The parameters of the action you want to take. Must be valid JSON. 
        The action_params should be valid for that action.
        The action_params should be a dictionary with the parameters of the action.
            {
            "param1": "value1",
            "param2": "value2"
        }
}
```
"""

# format_instruction = f"""Only output the json object. Do not output anything other than the json object which should be directly parsable by pydantic. Output the json object in the following format:\n```json\n{(Response.model_json_schema())}```\n"""


next_action = """You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page and the task in hand before deciding on your next action."""

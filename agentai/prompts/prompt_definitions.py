import json

from computergym.actions.action import ClickAction, InputText, action_examples

from .utils import Response

click_example_response = Response(
    observation="The current page looks like a homepage with signin button relevant for this task.",
    reasoning="I need to click on the Submit button to send the form. I will use the click action on the button, which has bid 12.",
    action_name=ClickAction.__name__,
    action_params=action_examples[ClickAction.__name__].model_dump(),
)

input_text_example_response = Response(
    observation="The current page looks signinpage with email and password fields relevant for this task.",
    reasoning="I need to enter text in text field with bid 12. I will use the input_text action.",
    action_name=InputText.__name__,
    action_params=action_examples[InputText.__name__].model_dump(),
)

instruction_prompt = """You are a UI Assistant, your goal is to help the user perform tasks using a web browser. You can
communicate with the user via a chat, to which the user gives you instructions and to which you
can send back messages. You have access to a web browser that both you and the user can see,
and with which only you can interact via specific commands.

Review the instructions from the user, the current state of the page and all other information
to find the best possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions. Try not repeating same actions on same page as that would not be helpful to the user."""


response_dict = {
    property: value["description"]
    for property, value in Response.model_json_schema()["properties"].items()
}
format_instruction = f"""
Only output the json object. Do not output anything other than the json object which should be directly parsable by pydantic. Output the json object in the following format:
```json
{json.dumps(response_dict, indent=4)}
```
"""


next_action = """You will now think step by step and produce your next best action. Reflect on your past actions, any resulting error message, the current state of the page and the task in hand before deciding on your next action."""

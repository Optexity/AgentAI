import json

from computergym import Observation
from pydantic import BaseModel

from .prompt_definitions import (
    click_example_response,
    format_instruction,
    input_text_example_response,
    instruction_prompt,
    next_action,
)
from .utils import PromptKeys, PromptStyle, Response, style


def custom_json_schema(action_function: type[BaseModel]) -> dict:
    schema = action_function.model_json_schema()
    return {
        "action_name": action_function.__name__,
        "action_description": schema.get("description", action_function.__name__),
        "action_params": {
            field: {
                "param_description": schema["properties"][field]["description"],
                "param_type": schema["properties"][field]["type"],
            }
            for field in schema["properties"]
        },
    }


def get_action_space_prompt(action_space: list[BaseModel]) -> str:
    separator = style[PromptKeys.AVAILABLE_ACTIONS][PromptStyle.LIST_SEPARATOR]
    prompt = ""
    for i, action in enumerate(action_space):
        prompt += (
            f"{separator} {i}\n{json.dumps(custom_json_schema(action),indent=4)}\n"
        )
    return prompt


def get_example_response_prompt() -> str:
    separator = style[PromptKeys.EXAMPLE_RESPONSE][PromptStyle.LIST_SEPARATOR]
    examples = [
        click_example_response.model_dump(),
        input_text_example_response.model_dump(),
    ]
    prompt = ""
    for i, response in enumerate(examples):
        prompt += f"{separator} {i}\n```json\n{json.dumps(response,indent=4)}\n```\n"
    return prompt


def get_previous_response_prompt(response_history: list[Response]) -> str:
    separator = style[PromptKeys.PREVIOUS_RESPONSES][PromptStyle.LIST_SEPARATOR]
    prompt = ""
    for i, response in enumerate(response_history):
        prompt += (
            f"{separator} {i}\n```json\n{response.model_dump_json(indent=4)}\n```\n"
        )
    return prompt


def get_system_prompt(keys: list[PromptKeys], action_space: list[BaseModel]) -> str:
    prompt = ""
    if PromptKeys.INSTRUCTION in keys:
        st = style[PromptKeys.INSTRUCTION]
        prompt += (
            f"{st[PromptStyle.BEGIN]}\n{instruction_prompt}\n{st[PromptStyle.END]}\n\n"
        )

    # if PromptKeys.RESPONSE_JSON_DESCRIPTION in keys:
    #     st = style[PromptKeys.RESPONSE_JSON_DESCRIPTION]
    #     prompt += f"{st[PromptStyle.BEGIN]}\n{json.dumps(Response.model_json_schema(), indent=4)}\n{st[PromptStyle.END]}\n\n"

    if PromptKeys.FORMAT_INSTRUCTION in keys:
        st = style[PromptKeys.FORMAT_INSTRUCTION]
        prompt += (
            f"{st[PromptStyle.BEGIN]}\n{format_instruction}\n{st[PromptStyle.END]}\n\n"
        )

    if PromptKeys.AVAILABLE_ACTIONS in keys:
        st = style[PromptKeys.AVAILABLE_ACTIONS]
        prompt += f"{st[PromptStyle.BEGIN]}\n{get_action_space_prompt(action_space)}\n{st[PromptStyle.END]}\n\n"

    if PromptKeys.EXAMPLE_RESPONSE in keys:
        st = style[PromptKeys.EXAMPLE_RESPONSE]
        prompt += f"{st[PromptStyle.BEGIN]}\n{get_example_response_prompt()}\n{st[PromptStyle.END]}\n\n"

    return prompt


def get_user_prompt(
    obs: Observation, response_history: list[Response], keys: list[PromptKeys]
) -> str:
    prompt = ""
    if PromptKeys.GOAL in keys:
        st = style[PromptKeys.GOAL]
        prompt += f"{st[PromptStyle.BEGIN]}\n{obs.goal}\n{st[PromptStyle.END]}\n\n"

    if PromptKeys.CURRENT_OBSERVATION in keys:
        st = style[PromptKeys.CURRENT_OBSERVATION]
        prompt += f"{st[PromptStyle.BEGIN]}\n{st[PromptStyle.DESCRIPTION]}\n{obs.axtree}\n{st[PromptStyle.END]}\n\n"

    if PromptKeys.PREVIOUS_RESPONSES in keys:
        st = style[PromptKeys.PREVIOUS_RESPONSES]
        prompt += f"{st[PromptStyle.BEGIN]}\n{get_previous_response_prompt(response_history)}\n{st[PromptStyle.END]}\n\n"

    if PromptKeys.PREVIOUS_ACTION_ERROR in keys and obs.last_action_error:
        st = style[PromptKeys.PREVIOUS_ACTION_ERROR]
        prompt += f"""{st[PromptStyle.BEGIN]}\n{obs.last_action_error}\n{st[PromptStyle.END]}\n\n"""

    if PromptKeys.NEXT_STEP in keys:
        st = style[PromptKeys.NEXT_STEP]
        prompt += f"{st[PromptStyle.BEGIN]}\n{next_action}\n{st[PromptStyle.END]}\n\n"

    return prompt

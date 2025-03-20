import json
import logging

from computergym import ActionTypes, Observation, OpenEndedWebsite, get_action_signature
from computergym.actions.action import ActionTypes
from pydantic import BaseModel

from .models import GeminiModels, VLLMModels, get_llm_model
from .prompts.prompts import system_prompt, user_prompt
from .prompts.utils import PromptKeys, PromptStyle, Response, Roles, style
from .utils import response_to_action

logger = logging.getLogger(__name__)


def get_action_prompt(action_type: ActionTypes) -> dict:
    name = action_type.value
    description = get_action_signature(action_type)
    description["action_name"] = name
    return description


def get_action_space_prompt(action_space: list[ActionTypes]) -> str:
    separator = style[PromptKeys.AVAILABLE_ACTIONS][PromptStyle.LIST_SEPARATOR]
    prompt = ""
    for i, action in enumerate(action_space):
        prompt += f"{separator} {i}\n{json.dumps(get_action_prompt(action),indent=4)}\n"
    return prompt


def get_example_response_prompt() -> str:
    separator = style[PromptKeys.EXAMPLE_RESPONSE][PromptStyle.LIST_SEPARATOR]
    prompt = ""
    for i, response in enumerate(system_prompt[PromptKeys.EXAMPLE_RESPONSE]):
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


def get_system_prompt(keys: list[PromptKeys], action_space: list[ActionTypes]) -> str:
    prompt = ""
    if PromptKeys.INSTRUCTION in keys:
        st = style[PromptKeys.INSTRUCTION]
        prompt += f"{st[PromptStyle.BEGIN]}\n{system_prompt[PromptKeys.INSTRUCTION]}\n{st[PromptStyle.END]}\n\n"

    # if PromptKeys.RESPONSE_JSON_DESCRIPTION in keys:
    #     st = style[PromptKeys.RESPONSE_JSON_DESCRIPTION]
    #     prompt += f"{st[PromptStyle.BEGIN]}\n{json.dumps(system_prompt[PromptKeys.RESPONSE_JSON_DESCRIPTION], indent=4)}\n{st[PromptStyle.END]}\n\n"

    if PromptKeys.FORMAT_INSTRUCTION in keys:
        st = style[PromptKeys.FORMAT_INSTRUCTION]
        prompt += f"{st[PromptStyle.BEGIN]}\n{system_prompt[PromptKeys.FORMAT_INSTRUCTION]}\n{st[PromptStyle.END]}\n\n"

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
        prompt += f"{st[PromptStyle.BEGIN]}\n{user_prompt[PromptKeys.NEXT_STEP]}\n{st[PromptStyle.END]}\n\n"

    return prompt


class BasicAgent:
    def __init__(
        self,
        model_name: GeminiModels | VLLMModels,
        env: OpenEndedWebsite,
        use_instructor: bool = False,
        port: int = None,
    ):
        self.model_name = model_name
        self.use_instructor = use_instructor
        self.action_space = env.get_action_space()
        self.system_prompt = get_system_prompt(
            [
                PromptKeys.INSTRUCTION,
                PromptKeys.RESPONSE_JSON_DESCRIPTION,
                PromptKeys.FORMAT_INSTRUCTION,
                PromptKeys.AVAILABLE_ACTIONS,
                PromptKeys.EXAMPLE_RESPONSE,
            ],
            self.action_space,
        )
        self.response_history: list[Response] = []

        self.model = get_llm_model(model_name, use_instructor=use_instructor, port=port)

    def get_input_messages(self, obs: Observation) -> list[dict]:
        keys = [
            PromptKeys.GOAL,
            PromptKeys.CURRENT_OBSERVATION,
            PromptKeys.PREVIOUS_ACTION_ERROR,
            PromptKeys.NEXT_STEP,
            PromptKeys.PREVIOUS_RESPONSES,
        ]
        messages = [
            {"role": Roles.SYSTEM, "content": self.system_prompt},
            {
                "role": Roles.USER,
                "content": get_user_prompt(obs, self.response_history, keys),
            },
        ]
        return messages

    def get_next_action(self, obs: Observation) -> tuple[Response, BaseModel]:
        input_messages = self.get_input_messages(obs)
        model_response = self.model.get_model_response(input_messages)
        self.response_history.append(model_response)
        action = response_to_action(model_response)
        return model_response, action

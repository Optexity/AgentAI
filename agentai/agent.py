import json
import logging

from computergym import (
    ActionTypes,
    ObsProcessorTypes,
    OpenEndedWebsite,
    get_action_object,
    get_action_signature,
)
from computergym.actions.action import ActionTypes
from models import LLMModelType, get_llm_model
from prompts.prompts import system_prompt, user_prompt
from prompts.utils import PromptKeys, PromptStyle, Response, Roles, style
from pydantic import BaseModel

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
        prompt += f"{separator} {i}\n{json.dumps(response,indent=4)}\n"
    return prompt


def get_system_prompt(keys: list[PromptKeys], action_space: list[ActionTypes]) -> str:
    prompt = ""
    if PromptKeys.INSTRUCTION in keys:
        st = style[PromptKeys.INSTRUCTION]
        prompt += f"{st[PromptStyle.BEGIN]}\n{system_prompt[PromptKeys.INSTRUCTION]}\n{st[PromptStyle.END]}\n\n"

    if PromptKeys.RESPONSE_JSON_DESCRIPTION in keys:
        st = style[PromptKeys.RESPONSE_JSON_DESCRIPTION]
        prompt += f"{st[PromptStyle.BEGIN]}\n{json.dumps(system_prompt[PromptKeys.RESPONSE_JSON_DESCRIPTION], indent=4)}\n{st[PromptStyle.END]}\n\n"

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


def get_user_prompt(obs: dict, keys: list[PromptKeys]) -> str:
    prompt = ""
    if PromptKeys.GOAL in keys:
        st = style[PromptKeys.GOAL]
        prompt += f"{st[PromptStyle.BEGIN]}\n{obs['chat_messages'][-1]['message']}\n{st[PromptStyle.END]}\n\n"

    if PromptKeys.CURRENT_OBSERVATION in keys:
        st = style[PromptKeys.CURRENT_OBSERVATION]
        prompt += f"{st[PromptStyle.BEGIN]}\n{st[PromptStyle.DESCRIPTION]}\n{obs[ObsProcessorTypes.axtree]}\n{st[PromptStyle.END]}\n\n"

    if PromptKeys.PREVIOUS_ACTION_ERROR in keys and obs["last_action_error"]:
        st = style[PromptKeys.PREVIOUS_ACTION_ERROR]
        prompt += f"""{st[PromptStyle.BEGIN]}\n{obs["last_action_error"]}\n{st[PromptStyle.END]}\n\n"""

    if PromptKeys.NEXT_STEP in keys:
        st = style[PromptKeys.NEXT_STEP]
        prompt += f"{st[PromptStyle.BEGIN]}\n{user_prompt[PromptKeys.NEXT_STEP]}\n{st[PromptStyle.END]}\n\n"

    return prompt


class BasicAgent:
    def __init__(self, name: str, env: OpenEndedWebsite, agent_description: str):
        self.name = name
        self.agent_description = agent_description
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

        # self.model = get_llm_model("models/gemini-2.0-flash", LLMModelType.GEMINI)
        self.model = get_llm_model(
            "meta-llama/Meta-Llama-3.1-8B-Instruct", LLMModelType.LLAMA_FACTORY_VLLM
        )

    def get_history_messages(self) -> list[dict]:
        return
        messages = []
        for i, response in enumerate(self.response_history):
            messages.append({"role": "user", "content": next_action})
            messages.append(
                {"role": "assistant", "content": response.model_dump_json(indent=4)}
            )
        return messages

    def get_input_messages(self, obs: dict) -> list[dict]:
        keys = [
            PromptKeys.GOAL,
            PromptKeys.CURRENT_OBSERVATION,
            PromptKeys.PREVIOUS_ACTION_ERROR,
            PromptKeys.NEXT_STEP,
        ]
        messages = [
            {"role": Roles.SYSTEM, "content": self.system_prompt},
            {"role": Roles.USER, "content": get_user_prompt(obs, keys)},
        ]
        return messages

    def parse_model_response(self, model_response: Response) -> BaseModel:
        action_name = model_response.action_name
        action_params = model_response.action_params
        action_type = ActionTypes[action_name]
        action_object = get_action_object(action_type)
        action = action_object.model_validate(action_params)
        return action

    def get_next_action(self, obs: str) -> tuple[Response, BaseModel]:
        print("here1")
        input_messages = self.get_input_messages(obs)
        print("here2")
        model_response = self.model.get_model_response(input_messages)
        print("here3")
        self.response_history.append(model_response)
        print("here4")
        action = self.parse_model_response(model_response)
        print("here5")
        return model_response, action

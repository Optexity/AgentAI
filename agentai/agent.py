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
from PIL import Image
from prompts import (
    Response,
    example_actions,
    instruction_prompt,
    next_action,
    trajectories,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def get_action_prompt(action_type: ActionTypes) -> dict:
    name = action_type.value
    description = get_action_signature(action_type)
    description["action_name"] = name
    return description


def get_action_space_prompt(action_space: list[ActionTypes]) -> str:
    prompt = ""
    for i, action in enumerate(action_space):
        prompt += f"Action {i}\n{json.dumps(get_action_prompt(action),indent=4)}\n"
    return prompt


def get_system_prompt(action_space: list[ActionTypes]) -> str:
    prompt = f"""
    # Instructions:
    {instruction_prompt}

    # Available actions:
    {get_action_space_prompt(action_space)}

    # Example actions:
    {example_actions}
    """

    trajectories_str = f"""\n
    # Example trajectories:
     {trajectories}
    """
    prompt += trajectories_str
    return prompt


def get_som_prompt(obs: dict) -> str:

    if ObsProcessorTypes.som in obs:

        pil_image = Image.fromarray(obs[ObsProcessorTypes.som])
        content = [
            "Current screenshot of the webpage. Use the number on the items to extract bid. The numbers are the IDs of the elements in the AXTree.\n",
            pil_image,
        ]
        return [{"role": "user", "content": content}]
    return []


def get_axtree_prompt(obs: dict) -> str:

    if ObsProcessorTypes.axtree in obs:
        content = [
            f"""Current AXTree of the webpage. Use the number on the items to extract bid.\n""",
            f"{obs[ObsProcessorTypes.axtree]}\n",
        ]
        return [{"role": "user", "content": content}]
    return []


def get_final_prompt(obs: dict):
    prompt = f"Task: {next_action}\n{obs['chat_messages'][-1]['message']}"
    return prompt


def get_last_error(obs: dict) -> str:
    if obs["last_action_error"] is None:
        return []
    content = f"""
    # Previous error message
    {obs['last_action_error']}
    """
    return [{"role": "user", "content": content}]


class BasicAgent:
    def __init__(self, name: str, env: OpenEndedWebsite, agent_description: str):
        self.name = name
        self.agent_description = agent_description
        self.action_space = env.get_action_space()
        self.system_prompt = get_system_prompt(self.action_space)
        self.response_history: list[Response] = []

        self.model = get_llm_model("models/gemini-2.0-flash", LLMModelType.GEMINI)

    def get_history_messages(self) -> list[dict]:
        messages = []
        for i, response in enumerate(self.response_history):
            messages.append({"role": "user", "content": next_action})
            messages.append(
                {"role": "assistant", "content": response.model_dump_json(indent=4)}
            )
        return messages

    def get_user_input(self) -> str:
        user_input = input("Enter your input: ").lower().strip()
        if user_input.lower() == "":
            return []
        return [{"role": "user", "content": user_input}]

    def get_input_messages(self, obs: dict) -> list[dict]:
        messages = (
            [{"role": "system", "content": self.system_prompt}]
            + self.get_history_messages()
            + get_axtree_prompt(obs)
            # + get_som_prompt(obs)
            # + self.get_user_input()
            + get_last_error(obs)
            + [{"role": "user", "content": get_final_prompt(obs)}]
        )

        return messages

    def parse_model_response(self, model_response: Response) -> BaseModel:
        action_name = model_response.action_name
        action_params = model_response.action_params
        action_type = ActionTypes[action_name]
        action_object = get_action_object(action_type)
        action = action_object.model_validate(action_params)
        return action

    def get_next_action(self, obs: str) -> tuple[Response, BaseModel]:
        input_messages = self.get_input_messages(obs)
        model_response = self.model.get_model_response(input_messages)
        self.response_history.append(model_response)
        action = self.parse_model_response(model_response)
        return model_response, action

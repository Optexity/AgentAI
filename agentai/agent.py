import json
import logging
import os

import google.generativeai as genai
import instructor
from computergym import (
    ActionTypes,
    ObsProcessorTypes,
    OpenEndedWebsite,
    get_action_object,
    get_action_signature,
)
from computergym.actions.action import ActionTypes
from PIL import Image
from prompts import Response, example_actions, instruction_prompt, next_action
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
    return prompt


def get_user_prompt(obs: dict, obs_type: ObsProcessorTypes) -> str:

    if obs_type == ObsProcessorTypes.som:
        pil_image = Image.fromarray(obs[ObsProcessorTypes.som])
        return [
            "Current screenshot of the webpage. Use the number on the items to extract bid",
            pil_image,
        ]

    if obs[ObsProcessorTypes.axtree]:
        return f"""Current AXTree of the webpage. Use the number on the items to extract bid.\n{obs[ObsProcessorTypes.axtree]}"""


def get_final_prompt(obs: dict):
    prompt = f"Task: {next_action}\n{obs['chat_messages'][-1]['message']}"
    return prompt


class BasicAgent:
    def __init__(self, name: str, env: OpenEndedWebsite, agent_description: str):
        self.name = name
        self.agent_description = agent_description
        self.action_space = env.get_action_space()
        self.system_prompt = get_system_prompt(self.action_space)
        self.response_history: list[Response] = []

        self.llm_model_name = "models/gemini-2.0-flash"

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name=self.llm_model_name),
            mode=instructor.Mode.GEMINI_JSON,
        )

    def get_history_messages(self) -> list[dict]:
        messages = []
        for i, response in enumerate(self.response_history):
            messages.append({"role": "user", "content": next_action})
            messages.append(
                {"role": "assistant", "content": response.model_dump_json(indent=4)}
            )
        return messages

    def get_model_response(self, obs: dict) -> Response:
        response: Response = self.client.create(
            response_model=Response,
            messages=[{"role": "system", "content": self.system_prompt}]
            + self.get_history_messages()
            + [{"role": "user", "content": get_user_prompt(obs, ObsProcessorTypes.som)}]
            + [{"role": "user", "content": get_final_prompt(obs)}],
        )

        return response

    def parse_model_response(self, model_response: Response) -> BaseModel:
        action_name = model_response.action_name
        action_params = model_response.action_params
        action_type = ActionTypes[action_name]
        action_object = get_action_object(action_type)
        action = action_object.model_validate(action_params)
        return action

    def get_next_action(self, obs: str) -> tuple[Response, BaseModel]:
        model_response = self.get_model_response(obs)
        self.response_history.append(model_response)
        action = self.parse_model_response(model_response)
        return model_response, action

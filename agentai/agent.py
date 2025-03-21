import logging

from agentai.models import GeminiModels, VLLMModels, get_llm_model
from agentai.prompts import (
    PromptKeys,
    Response,
    Roles,
    get_system_prompt,
    get_user_prompt,
)
from agentai.utils import response_to_action
from computergym import Observation, OpenEndedWebsite
from pydantic import BaseModel

logger = logging.getLogger(__name__)


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
        # TODO: change this to chat style so that it can be cached
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

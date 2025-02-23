import json
import os
import re

import google.generativeai as genai
import instructor
from computergym import (
    ActionTypes,
    ObsProcessorTypes,
    OpenEndedWebsite,
    get_action_object,
    get_action_signature,
)
from prompts import example_actions, format_instruction, instruction_prompt, next_action
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


def get_action_prompt(action_type: ActionTypes) -> str:
    name = action_type.value
    description = get_action_signature(action_type)
    description["action_name"] = name
    return description


def get_action_space_prompt(action_space: list[ActionTypes]) -> str:
    prompt = ""
    for i, action in enumerate(action_space):
        prompt += f"Action {i}\n{get_action_prompt(action)}\n"
    return prompt


def get_system_prompt(action_space: list[ActionTypes]) -> str:
    prompt = f"""
    # Instructions:
    {instruction_prompt}

    # Available actions:
    {get_action_space_prompt(action_space)}

    # Example actions:
    {example_actions}
    
    {next_action}
    """
    return prompt


def get_user_prompt(obs: dict) -> str:
    prompt = f"""
    # AXTree Observation:
    {obs[ObsProcessorTypes.axtree]}

    # Task:
    {obs['chat_messages'][-1]['message']}
    """
    return prompt


class BasicAgent:
    def __init__(self, name: str, env: OpenEndedWebsite, agent_description: str):
        self.name = name
        self.agent_description = agent_description
        self.action_space = env.get_action_space()
        self.system_prompt = get_system_prompt(self.action_space)

        self.llm_model_name = "models/gemini-2.0-flash"

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.client = instructor.from_gemini(
            client=genai.GenerativeModel(
                model_name="models/gemini-2.0-flash",  # model defaults to "gemini-pro"
            ),
            mode=instructor.Mode.GEMINI_JSON,
        )

    def get_model_response(self, obs: dict) -> Response:
        response: Response = self.client.create(
            response_model=Response,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": get_user_prompt(obs)},
            ],
        )
        return response

    def parse_model_response(self, model_response: Response) -> BaseModel:
        action_name = model_response.action_name
        action_params = model_response.action_params
        action_type = ActionTypes[action_name]
        action_object = get_action_object(action_type)
        action = action_object.model_validate(action_params)
        return action

    def get_next_action(self, obs: str) -> BaseModel:
        model_response = self.get_model_response(obs)
        action = self.parse_model_response(model_response)
        return action

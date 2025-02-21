import re

import gymnasium as gym
from computergym import (
    ActionTypes,
    EnvTypes,
    ObsProcessorTypes,
    OpenEndedWebsite,
    get_action_description,
    get_parameters_description,
)
from litellm import completion
from prompts import example_actions, format_instruction, instruction_prompt, next_action


def get_action_prompt(action_type: ActionTypes) -> str:
    name = action_type.value
    description = get_action_description(action_type)
    parameters = get_parameters_description(action_type)
    prompt = f"""## Action Name: {name}. Description {description}. Parameters: {parameters}"""
    return prompt


def get_action_space_prompt(action_space: list[ActionTypes]) -> str:
    prompt = ""
    for action in action_space:
        prompt += f"{get_action_prompt(action)}\n"
    return prompt


def get_system_prompt(action_space: list[ActionTypes]) -> str:
    prompt = f"""
    # Instructions:
    {instruction_prompt}

    # Available actions:
    {get_action_space_prompt(action_space)}

    # Example actions:
    {example_actions}

    # Format:
    {format_instruction}
    
    {next_action}
    """
    return prompt


def get_user_prompt(obs: dict, task: str) -> str:
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

    def get_model_response(self, obs: dict) -> str:
        response = completion(
            model="gemini/gemini-2.0-flash",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": get_user_prompt(obs, "signin")},
            ],
        )
        return response.choices[0].message.content

    def parse_model_response(self, response: str) -> tuple[ActionTypes, list[str]]:
        try:
            pattern = r"\`\`\`(\w+)\((.*?)\)\`\`\`"
            match = re.search(pattern, response)
            if match:
                action_type = ActionTypes[match.group(1)]
                action_params = []
                for a in match.group(2).split(","):
                    param = a.strip()
                    if param.startswith('"'):
                        param = param[1:]
                    if param.endswith('"'):
                        param = param[:-1]
                    action_params.append(param)
                return action_type, action_params
        except Exception as e:
            print(f"Error parsing model response: {e}")
            pass

        return None, None

    def get_next_action(self, obs: str) -> tuple[ActionTypes, list[str]]:
        model_response = self.get_model_response(obs)
        action_type, action_params = self.parse_model_response(model_response)
        return action_type, action_params

    def act(self, action: str):
        print(f"{self.name} is performing action: {action}")

    def respond(self, response: str):
        print(f"{self.name} responds with: {response}")

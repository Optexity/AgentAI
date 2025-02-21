import gymnasium as gym
from computergym import ActionTypes, EnvTypes, ObsProcessorTypes, OpenEndedWebsite


class BasicAgent:
    def __init__(self, name: str, env: OpenEndedWebsite, description: str):
        self.name = name
        self.description = description
        self.action_space = env.get_action_space()

    def act(self, action: str):
        print(f"{self.name} is performing action: {action}")

    def respond(self, response: str):
        print(f"{self.name} responds with: {response}")

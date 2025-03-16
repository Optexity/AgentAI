import argparse
import ast
import json
import os

from agent import BasicAgent
from agentai.agent import BasicAgent
from agentai.models import GeminiModels, VLLMModels
from computergym import (
    BrowserEnvTypes,
    EnvTypes,
    ObsProcessorTypes,
    OpenEndedWebsite,
    make_env,
)
from computergym.demonstrations import Demonstration


def main(input_file: str, output_file: str):
    demonstrations = Demonstration.from_json(input_file)
    env: OpenEndedWebsite = make_env(
        "random",
        EnvTypes.browser,
        BrowserEnvTypes.workarena,
        [ObsProcessorTypes.axtree],
        headless=True,
    )
    agent = BasicAgent(GeminiModels.GEMINI_2_0_FLASH, env, False, None)
    data = []
    for demonstration in demonstrations:
        input_messages = agent.get_input_messages(demonstration.obs)
        output_action = demonstration.action
    with open(output_file, "w") as f:
        json.dump(data, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data for SFT")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

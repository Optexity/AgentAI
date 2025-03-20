import argparse
import os

import yaml
from agent import BasicAgent
from computergym import BrowserEnvTypes, EnvTypes, OpenEndedWebsite, make_env
from computergym.envs.browser import History
from models import GeminiModels
from utils import action_to_response

SAVE_DIR = "save_dir"
TASKS = "tasks"
TASK_NAME = "task_name"
DESCRIPTION = "description"
URL = "url"
PROCESSED_OUTPUT_DIR = "processed_output_dir"


def read_file(file_path):
    with open(file_path, "r") as f:
        data = f.read().strip()
    return data


def get_input_output(env: OpenEndedWebsite, processed_output_dir: str):

    full_data = []

    agent = BasicAgent(GeminiModels.GEMINI_2_0_FLASH, env, False)

    history_list = History.read_history(processed_output_dir)

    for history in history_list:
        messages = agent.get_input_messages(history.obs)
        system_message = messages[0]["content"]
        user_message = messages[1]["content"]
        target_response = action_to_response(history.action)
        target = f"```json\n{target_response.model_dump_json(indent=4)}\n```"

        # TODO: augment so that agent can learn with and without history
        agent.response_history.append(target_response)
        full_data.append(
            {
                "system": system_message,
                "instruction": user_message,
                "input": "",
                "output": target,
            }
        )


def main(yaml_file_path: str):
    with open(yaml_file_path, "r") as file:
        data = yaml.safe_load(file)

    env: OpenEndedWebsite = make_env(
        None,
        EnvTypes.browser,
        BrowserEnvTypes.openended,
        cache_dir=None,
        goal_message=None,
        headless=True,
    )

    for task in data[TASKS]:
        task_name = task[TASK_NAME]
        env.goal = task[DESCRIPTION]
        env.url = task[URL]
        processed_output_dir = os.path.join(
            data[SAVE_DIR], task_name, data[PROCESSED_OUTPUT_DIR]
        )
        get_input_output(env, processed_output_dir)


if __name__ == "__main__":
    main("/Users/sankalp/repository/github/AWS_DATA/trajectorybucket/trace_profiling")
    exit()
    parser = argparse.ArgumentParser(description="Prepare Lamma Factory Data")
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()
    main()

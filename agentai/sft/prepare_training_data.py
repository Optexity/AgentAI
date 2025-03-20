import argparse
import json
import os

import yaml
from agentai.agent import BasicAgent
from agentai.models import GeminiModels
from agentai.utils import action_to_response
from computergym import BrowserEnvTypes, EnvTypes, OpenEndedWebsite, make_env
from computergym.envs.browser import History

SAVE_DIR = "save_dir"
TASKS = "tasks"
TASK_NAME = "task_name"
DESCRIPTION = "description"
URL = "url"
PROCESSED_OUTPUT_DIR = "processed_output_dir"


def get_input_output(env: OpenEndedWebsite, processed_output_dir: str):

    task_data = []
    agent = BasicAgent(GeminiModels.GEMINI_2_0_FLASH, env, False)
    history_list = History.read_history(processed_output_dir)

    for history in history_list:
        history.obs.goal = env.goal
        history.obs.url = env.url
        messages = agent.get_input_messages(history.obs)
        system_message = messages[0]["content"]
        user_message = messages[1]["content"]
        target_response = action_to_response(history.action)
        target = f"```json\n{target_response.model_dump_json(indent=4)}\n```"

        # TODO: augment so that agent can learn with and without history
        agent.response_history.append(target_response)
        task_data.append(
            {
                "system": system_message,
                "instruction": user_message,
                "input": "",
                "output": target,
            }
        )
    return task_data


def main(yaml_file_path: str, save_dir: str):
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

    all_data = []
    for task in data[TASKS]:
        task_name = task[TASK_NAME]
        env.goal = task[DESCRIPTION]
        env.url = task[URL]
        processed_output_dir = os.path.join(
            data[SAVE_DIR], task_name, data[PROCESSED_OUTPUT_DIR]
        )
        task_data = get_input_output(env, processed_output_dir)
        all_data.extend(task_data)

    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, "training_data.json")
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Lamma Factory Data")
    parser.add_argument("--yaml", type=str, help="Input file path")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()
    main(args.yaml, args.output)

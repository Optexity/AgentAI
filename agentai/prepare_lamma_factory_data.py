import argparse
import ast
import json
import os
import re

from agent import BasicAgent
from browsergym.workarena import SERVICE_CATALOG_TASKS
from browsergym.workarena.tasks.base import AbstractServiceNowTask
from computergym import (
    BrowserEnvTypes,
    EnvTypes,
    ObsProcessorTypes,
    OpenEndedWebsite,
    make_env,
)
from computergym.utils import save_str_obs
from prompts.utils import Response
from utils import get_logger


def read_file(file_path):
    with open(file_path, "r") as f:
        data = f.read().strip()
    return data


def read_action(file_path):
    data = read_file(file_path)
    action_name, action_params = data.split("action_params=")
    action_name = ast.literal_eval(action_name.split("action_name=")[1])

    if "options" in action_params and "'options'" not in action_params:
        action_params = action_params.replace("options", "'options'")

    action_params = ast.literal_eval(action_params)

    response = Response.model_validate(
        {"action_name": action_name, "action_params": action_params}
    )
    return response


def main(input_dir: str):
    env: OpenEndedWebsite = make_env(
        "random",
        EnvTypes.browser,
        BrowserEnvTypes.workarena,
        [ObsProcessorTypes.axtree],
        headless=True,
    )
    agent = BasicAgent("basic_agent", env, "basic_agent")
    full_data = []
    for task_type in os.listdir(input_dir):
        task_path = os.path.join(input_dir, task_type)
        if not os.path.isdir(task_path):
            continue
        if task_type != "SERVICE_CATALOG_TASKS":
            continue
        for task_sub_type in os.listdir(task_path):
            task_sub_path = os.path.join(task_path, task_sub_type)
            if not os.path.isdir(task_sub_path):
                continue
            for seed in os.listdir(task_sub_path):
                seed_path = os.path.join(task_sub_path, seed)
                if not os.path.isdir(seed_path):
                    continue
                agent = BasicAgent("basic_agent", env, "basic_agent")
                goal = read_file(os.path.join(seed_path, "goal.txt"))
                all_steps = [a for a in os.listdir(seed_path) if a.startswith("step-")]
                for step in sorted(
                    all_steps, key=lambda x: int(x.removeprefix("step-"))
                ):
                    step_path = os.path.join(seed_path, step)
                    if not os.path.isdir(step_path):
                        continue
                    try:
                        action_response = read_action(
                            os.path.join(step_path, "action.txt")
                        )
                        action = agent.parse_model_response(action_response)
                        axtree = read_file(os.path.join(step_path, "axtree.txt"))
                        obs = {
                            ObsProcessorTypes.axtree: axtree,
                            ObsProcessorTypes.goal: goal,
                            ObsProcessorTypes.last_action_error: None,
                        }
                        messages = agent.get_input_messages(obs)
                        system_message = messages[0]["content"]
                        user_message = messages[1]["content"]
                        target = (
                            f"```json\n{action_response.model_dump_json(indent=4)}\n```"
                        )

                        agent.response_history.append(action_response)
                        full_data.append(
                            {
                                "system": system_message,
                                "instruction": user_message,
                                "output": target,
                            }
                        )
                    except Exception as e:
                        # print(os.path.join(step_path, "action.txt"))
                        # print(action_response)
                        print(f"Error in {step_path}: {e}")
                        import pdb

                        pdb.set_trace()
                        pass

    os.makedirs("./train_data/SERVICE_CATALOG_TASKS", exist_ok=True)
    with open(
        "./train_data/SERVICE_CATALOG_TASKS/service_catalog_tasks.json", "w"
    ) as f:
        json.dump(full_data, f, indent=4)


if __name__ == "__main__":
    main(
        "/Users/sankalp/repository/github/Reinforce-Align-AI/trajectorybucket/trace_profiling"
    )
    exit()
    parser = argparse.ArgumentParser(description="Prepare Lamma Factory Data")
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()
    main()

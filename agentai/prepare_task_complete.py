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
from tqdm import tqdm
from utils import get_logger


def convert_task_name(name: str):
    name = name.removesuffix("Task")  # Remove "Task" at the end
    name = re.sub(
        r"([a-z])([A-Z])", r"\1-\2", name
    )  # Insert '-' before middle capital letters
    name = re.sub(
        r"([A-Z])([A-Z])", r"\1-\2", name
    )  # Handle consecutive uppercase letters
    return name.lower()  # Convert to lowercase


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


def main2(
    task_entrypoint: callable,
    input_dir: str,
    task_type: str,
    task_sub_type: str,
    seed: int,
):
    env: OpenEndedWebsite = make_env(
        "random",
        EnvTypes.browser,
        BrowserEnvTypes.workarena,
        [ObsProcessorTypes.axtree, ObsProcessorTypes.html],
        headless=False,
    )
    agent = BasicAgent("basic_agent", env, "basic_agent", port=8000)

    task_path = os.path.join(input_dir, task_type)
    task_sub_path = os.path.join(task_path, task_sub_type)
    seed_path = os.path.join(task_sub_path, f"seed-{seed}")
    print(seed_path)
    sequences = []

    goal = read_file(os.path.join(seed_path, "goal.txt"))
    all_steps = [a for a in os.listdir(seed_path) if a.startswith("step-")]
    for step in sorted(all_steps, key=lambda x: int(x.removeprefix("step-"))):
        step_path = os.path.join(seed_path, step)
        if not os.path.isdir(step_path):
            continue
        try:
            action_response = read_action(os.path.join(step_path, "action.txt"))
            action = agent.parse_model_response(action_response)
            axtree = read_file(os.path.join(step_path, "axtree.txt"))
            sequences.append({"action": action, "axtree": axtree})
        except Exception as e:
            print(f"Error in {step_path}: {e}")
            import pdb

            pdb.set_trace()
            exit()
            pass

    if sequences[-1]["action"].__class__.__name__ == "TaskComplete":
        print(f"Task complete: {task_sub_type} {seed}")
        return

    for i in tqdm(range(100)):
        task: AbstractServiceNowTask = task_entrypoint(seed=i)
        if task.setup_goal(None)[0] == goal:
            print(f"Seed {i} matches")
            break
    import pdb

    pdb.set_trace()
    assert (
        goal == task.setup_goal(None)[0]
    ), f"Goal mismatch: {goal} != {task.setup_goal(None)[0]}"

    env: OpenEndedWebsite = make_env(
        "random",
        EnvTypes.browser,
        BrowserEnvTypes.workarena,
        [ObsProcessorTypes.axtree, ObsProcessorTypes.html],
        headless=False,
    )

    obs, info = env.reset()
    for step_num, step in enumerate(sequences):
        if step["axtree"].strip() != obs[ObsProcessorTypes.axtree].strip():
            import pdb

            pdb.set_trace()
        action = step["action"]
        obs, info = env.step(action)

    path = os.path.join(seed_path, f"step-{step_num+1}")
    print(path)

    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "action.txt"), "w") as f:
        f.write("action_name='task_complete' action_params={'msg': 'Task complete!'}")
    with open(os.path.join(path, "axtree.txt"), "w") as f:
        f.write(obs[ObsProcessorTypes.axtree])
    with open(os.path.join(path, "html.txt"), "w") as f:
        f.write(obs[ObsProcessorTypes.html])


def main(args):

    task_entrypoint = SERVICE_CATALOG_TASKS[args.task_num]
    task_sub_type = convert_task_name(task_entrypoint.__name__)
    print(task_sub_type)
    main2(
        task_entrypoint, args.input, "SERVICE_CATALOG_TASKS", task_sub_type, args.seed
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Lamma Factory Data")
    parser.add_argument("--input", type=str, help="Input file path")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task_num", type=int, required=True)
    args = parser.parse_args()
    main(args)

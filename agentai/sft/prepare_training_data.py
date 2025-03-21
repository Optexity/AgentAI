import argparse
import json
import os

import pandas as pd
import yaml
from agentai.agent import BasicAgent
from agentai.models import GeminiModels
from agentai.utils import action_to_response
from computergym import BrowserEnvTypes, EnvTypes, OpenEndedWebsite, make_env
from computergym.envs.browser import History


def save_train_config(agent_config: dict, save_dir: str):

    train_config = agent_config["train_config"]
    train_config["model_name_or_path"] = agent_config["model_name_or_path"]
    train_config["output_dir"] = os.path.join(
        agent_config["adapter_name_or_path"],
        agent_config["agent_name"],
        agent_config["model_name_or_path"],
        train_config["finetuning_type"],
        train_config["stage"],
    )
    train_config["trust_remote_code"] = agent_config["trust_remote_code"]
    train_config["template"] = agent_config["template"]
    train_config["cutoff_len"] = agent_config["context_length"]
    train_config["dataset"] = agent_config["agent_name"]
    train_config["run_name"] = agent_config["agent_name"]

    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, "train_config.yaml")
    with open(output_file, "w") as f:
        yaml.safe_dump(train_config, f)


def save_inference_config(agent_config: dict, save_dir: str):
    train_config = agent_config["train_config"]
    inference_config = agent_config["inference_config"]
    inference_config["model_name_or_path"] = agent_config["model_name_or_path"]
    inference_config["adapter_name_or_path"] = train_config["output_dir"]
    inference_config["trust_remote_code"] = agent_config["trust_remote_code"]
    inference_config["template"] = agent_config["template"]
    inference_config["vllm_maxlen"] = agent_config["context_length"]

    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, "inference_config.yaml")
    with open(output_file, "w") as f:
        yaml.safe_dump(inference_config, f)


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


def save_llama_factory_data(all_data: list, save_dir: str):
    output_file = os.path.join(save_dir, "llama_factory_training_data.json")
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=4)


def save_gemini_data(all_data: list, save_dir: str):
    new_data = [
        {
            "input": f"""{item['system']}\n{item['instruction']}\n{item['input']}""",
            "output": item["output"],
        }
        for item in all_data
    ]
    new_data = pd.DataFrame(new_data)
    output_file = os.path.join(save_dir, "gemini_training_data.csv")
    new_data.to_csv(output_file, index=False, quoting=1)


def main(yaml_file_path: str):
    with open(yaml_file_path, "r") as file:
        agent_config = yaml.safe_load(file)

    with open(agent_config["html_data_config"], "r") as file:
        html_data_config = yaml.safe_load(file)

    env: OpenEndedWebsite = make_env(
        None,
        EnvTypes.browser,
        BrowserEnvTypes.openended,
        cache_dir=None,
        goal_message=None,
        headless=True,
    )

    all_data = []
    for task in html_data_config["tasks"]:
        task_name = task["task_name"]
        env.goal = task["description"]
        env.url = task["url"]
        processed_output_dir = os.path.join(
            html_data_config["save_dir"],
            task_name,
            html_data_config["processed_output_dir"],
        )
        task_data = get_input_output(env, processed_output_dir)
        all_data.extend(task_data)
    env.close()

    save_dir = os.path.join(agent_config["agent_dir"], agent_config["agent_name"])
    os.makedirs(save_dir, exist_ok=True)

    save_llama_factory_data(all_data, save_dir)
    save_gemini_data(all_data, save_dir)
    save_train_config(agent_config, save_dir)
    save_inference_config(agent_config, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Lamma Factory Data")
    parser.add_argument("--agent_config", type=str, required=True)
    args = parser.parse_args()
    main(args.agent_config)

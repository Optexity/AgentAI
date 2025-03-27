import argparse
import json
import os
import time

from agentai.agent import BasicAgent
from agentai.models import GeminiModels, VLLMModels
from agentai.utils import get_logger
from computergym import BrowserEnvTypes, EnvTypes, OpenEndedWebsite, make_env
from computergym.utils import save_str_to_file


def run(args):
    os.makedirs(args.log_path, exist_ok=True)

    logger = get_logger(
        __name__, log_path=args.log_path, log_to_console=args.log_to_console
    )

    env: OpenEndedWebsite = make_env(
        args.url,
        EnvTypes.browser,
        BrowserEnvTypes.openended,
        cache_dir=args.log_path,
        goal_message=args.goal,
        headless=args.headless,
        # proxy="http://38.154.227.167:5868",
        storage_state=args.storage_state,
    )
    if args.model == "vllm":
        agent = BasicAgent(VLLMModels.LLAMA_3_1_8B_INSTRUCT, env, False, args.port)
    else:
        agent = BasicAgent(GeminiModels.GEMINI_2_0_FLASH, env, False, args.port)

    obs, info = env.reset()

    while True:
        logger.info("-" * 20)
        logger.info(f"step: {env.current_step}")
        if env.current_step > 30:
            logger.info("Too many steps, stopping...")
            break
        model_response, action = agent.get_next_action(obs)

        if args.log_path:
            cache_dir = os.path.join(args.log_path, f"step-{env.current_step}")
            os.makedirs(cache_dir, exist_ok=True)
            string = model_response.model_dump()
            string = json.dumps(string, indent=4)
            save_str_to_file(
                string, cache_dir, f"model-response-{env.current_step}.txt"
            )

        logger.info(f"model_response: {model_response}")
        string = action.model_dump()
        string["action_name"] = action.__class__.__name__
        logger.info(f"action: {string}")
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the browser gym tasks.")
    parser.add_argument("--goal", type=str, required=True)
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--log_to_console", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--storage_state", type=str, default=None)
    parser.add_argument("--model", type=str, choices=["gemini", "vllm"], required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as e:
        print(f"Error: {e}")

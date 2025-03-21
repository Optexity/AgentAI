import argparse
import json
import os
import time

from agentai.agent import BasicAgent
from agentai.models import GeminiModels, VLLMModels
from agentai.utils import get_logger
from computergym import BrowserEnvTypes, EnvTypes, OpenEndedWebsite, make_env
from computergym.utils import save_str_to_file


def run(
    goal: str,
    url: str,
    log_path="./logs",
    log_to_console=False,
    headless=False,
    port=None,
):

    logger = get_logger(__name__, log_path=log_path, log_to_console=log_to_console)

    env: OpenEndedWebsite = make_env(
        url,
        EnvTypes.browser,
        BrowserEnvTypes.openended,
        cache_dir=log_path,
        goal_message=goal,
        headless=headless,
        # proxy="http://38.154.227.167:5868",
    )
    agent = BasicAgent(GeminiModels.TUNED_MODELS_HUBSPOT_V1, env, False, port)

    obs, info = env.reset()
    breakpoint()
    env.obs = env.get_obs()
    obs = env.obs

    while True:
        logger.info("-" * 20)
        logger.info(f"step: {env.current_step}")
        if env.current_step > 30:
            logger.info("Too many steps, stopping...")
            break
        model_response, action = agent.get_next_action(obs)

        if log_path:
            cache_dir = os.path.join(log_path, f"step-{env.current_step}")
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


def main(args):
    os.makedirs(args.log_path, exist_ok=True)
    run(
        args.goal,
        args.url,
        args.log_path,
        args.log_to_console,
        args.headless,
        args.port,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the browser gym tasks.")
    parser.add_argument("--goal", type=str, required=True)
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--log_to_console", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}")

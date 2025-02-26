import argparse
import json
import os

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
from computergym.utils import save_screenshot, save_str_obs
from tqdm import tqdm
from utils import get_logger


def run(task: AbstractServiceNowTask, log_path="./logs"):

    logger = get_logger(__name__, log_path=log_path, log_to_console=False)

    goal, _ = task.setup_goal(None)
    logger.info(f"Goal: {goal}")

    env: OpenEndedWebsite = make_env(
        task.start_url,
        EnvTypes.browser,
        BrowserEnvTypes.workarena,
        [
            ObsProcessorTypes.html,
            ObsProcessorTypes.axtree,
            ObsProcessorTypes.screenshot,
            ObsProcessorTypes.som,
        ],
        cache_dir=log_path,
        goal_message=goal,
        headless=False,
    )
    agent = BasicAgent("basic_agent", env, "basic_agent")

    obs, info = env.reset()

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
            save_str_obs(string, cache_dir, f"model-response-{env.current_step}.txt")

        logger.info(f"model_response: {model_response}")
        string = action.model_dump()
        string["action_name"] = action.__class__.__name__
        logger.info(f"action: {string}")
        obs, reward, terminated, truncated, info = env.step(action)
        logger.info(
            f"reward: {reward}, terminated: {terminated}, truncated: {truncated}"
        )
        if terminated or truncated:
            break
    # release the environment
    ## Validate
    reward, stop, message, info = task.validate(env.page, [])
    logger.info(f"Final Reward: {reward}")
    logger.info(f"Final Stop: {stop}")
    logger.info(f"Final Message: {message}")
    logger.info(f"Final Info: {info}")
    env.close()
    return reward


def main(args):
    task_entrypoint = SERVICE_CATALOG_TASKS[args.task_num]
    task = task_entrypoint(seed=args.seed)
    log_path = os.path.join(
        args.log_path, task_entrypoint.__name__, f"seed-{str(args.seed)}"
    )
    os.makedirs(log_path, exist_ok=True)
    reward = run(task, log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the browser gym tasks.")
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--task_num", type=int, required=True)
    args = parser.parse_args()
    main(args)

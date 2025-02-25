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
from tqdm import tqdm
from utils import get_logger


def run(task: AbstractServiceNowTask, log_path="./logs"):

    logger = get_logger(__name__, log_path=log_path)

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
    logger.info(f"Reward: {reward}, Stop: {stop}, Message: {message}, Info: {info}")
    env.close()
    return reward


def main():
    total_tasks = 0
    total_reward = 0
    for seed in tqdm(range(10)):
        for task_entrypoint in SERVICE_CATALOG_TASKS:
            task = task_entrypoint(seed=seed)
            log_path = os.path.join(
                "./logs", task_entrypoint.__name__, f"seed-{str(seed)}"
            )
            reward = run(task, log_path)
            total_reward += reward
            total_tasks += 1
    print(
        f"Total Score: {total_reward}, Total Tasks: {total_tasks}, Average Score: {total_reward / total_tasks}"
    )


if __name__ == "__main__":
    main()

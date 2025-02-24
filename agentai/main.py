from agent import BasicAgent
from browsergym.workarena import SERVICE_CATALOG_TASKS
from computergym import (
    BrowserEnvTypes,
    EnvTypes,
    ObsProcessorTypes,
    OpenEndedWebsite,
    make_env,
)
from utils import get_logger

logger = get_logger(__name__, log_path="./logs")


def main():
    task_entrypoint = SERVICE_CATALOG_TASKS[0]
    task = task_entrypoint(seed=0)
    goal, _ = task.setup_goal(None)

    print("Task:", task_entrypoint)
    print("Goal:", goal)

    env: OpenEndedWebsite = make_env(
        "lawyersaathi-v0",
        # "https://lawyersaathi.com",
        task.start_url,
        EnvTypes.browser,
        BrowserEnvTypes.workarena,
        [
            ObsProcessorTypes.html,
            ObsProcessorTypes.axtree,
            ObsProcessorTypes.screenshot,
            ObsProcessorTypes.som,
        ],
        cache_dir="./logs",
        goal_message=goal,
    )
    agent = BasicAgent("basic_agent", env, "basic_agent")

    obs, info = env.reset()

    while True:
        logger.info("-" * 20)
        logger.info(f"step: {env.current_step}")

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


if __name__ == "__main__":
    main()

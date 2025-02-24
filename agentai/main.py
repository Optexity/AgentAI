from pprint import pprint

from agent import BasicAgent
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
    env: OpenEndedWebsite = make_env(
        "lawyersaathi-v0",
        "https://lawyersaathi.com",
        # "https://dev283325.service-now.com/now/nav/ui/classic/params/target/catalog_home.do%3Fsysparm_view%3Dcatalog_default",
        EnvTypes.browser,
        BrowserEnvTypes.openended,
        [
            ObsProcessorTypes.html,
            ObsProcessorTypes.axtree,
            ObsProcessorTypes.screenshot,
            ObsProcessorTypes.som,
        ],
        cache_dir="./logs",
    )
    agent = BasicAgent("basic_agent", env, "basic_agent")

    obs, info = env.reset()
    action = None
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
    env.close()


if __name__ == "__main__":
    main()

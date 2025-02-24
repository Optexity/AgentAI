from pprint import pprint

from agent import BasicAgent
from computergym import (
    BrowserEnvTypes,
    EnvTypes,
    ObsProcessorTypes,
    OpenEndedWebsite,
    make_env,
)


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
        cache_dir="./cached_data",
    )
    agent = BasicAgent("basic_agent", env, "basic_agent")

    obs, info = env.reset()
    action = None
    while True:
        model_response, action = agent.get_next_action(obs)
        print("Model response:")
        pprint(model_response)
        print("Action:")
        pprint(action)
        # obs, reward, terminated, truncated, info = env.step(action)
        # action_type, action_params = agent.get_next_action(obs)
        # action = Action(action_type, action_params)
        # print(action_type)
        # print(action_params)
        # # obs, reward, terminated, truncated, info = env.step(action_type, action_params)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break
    # release the environment
    env.close()


if __name__ == "__main__":
    main()

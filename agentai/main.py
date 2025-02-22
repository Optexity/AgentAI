from agent import BasicAgent
from computergym import EnvTypes, ObsProcessorTypes, OpenEndedWebsite, make_env
from computergym.actions.action import Action


def main():
    env: OpenEndedWebsite = make_env(
        "lawyersaathi-v0",
        "https://lawyersaathi.com",
        EnvTypes.browser,
        [
            ObsProcessorTypes.html,
            ObsProcessorTypes.axtree,
            ObsProcessorTypes.screenshot,
            ObsProcessorTypes.som,
        ],
        cache_dir="./cached_data",
    )
    agent = BasicAgent("basic_agent", env, "basic_agent")

    obs, info = env.reset_()

    while True:
        action_type, action_params = agent.get_next_action(obs)
        action = Action(action_type, action_params)
        print(action_type)
        print(action_params)
        # obs, reward, terminated, truncated, info = env.step(action_type, action_params)
        obs, reward, terminated, truncated, info = env.step_(action)
        if terminated or truncated:
            break
    # release the environment
    env.close()


if __name__ == "__main__":
    main()

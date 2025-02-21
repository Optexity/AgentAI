import gymnasium as gym
from agent import BasicAgent
from computergym import ActionTypes, EnvTypes, ObsProcessorTypes, make_env


def get_action(obs: dict):
    return 0


def main():
    env = make_env(
        "lawyersaathi-v0",
        "https://lawyersaathi.com",
        EnvTypes.browser,
        [
            ObsProcessorTypes.html,
            ObsProcessorTypes.axtree,
            ObsProcessorTypes.screenshot,
            ObsProcessorTypes.som,
        ],
    )
    agent = BasicAgent("basic_agent", env, "basic_agent")

    obs, info = env.reset()

    while True:
        action_type, action_params = agent.get_next_action(obs)
        print(action_type)
        print(action_params)
        obs, reward, terminated, truncated, info = env.step(action_type, action_params)
        if terminated or truncated:
            break
    # release the environment
    env.close()


if __name__ == "__main__":
    main()

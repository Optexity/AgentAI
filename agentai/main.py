import gymnasium as gym
from computergym import ActionTypes, EnvTypes, ObsProcessorTypes, make_env


def get_action(obs: dict):
    return 0


def main():
    env = make_env(
        "lawyersaathi-v0",
        "https://lawyersaathi.com",
        EnvTypes.browser,
        [ObsProcessorTypes.html, ObsProcessorTypes.axtree],
    )

    obs, info = env.reset()

    import pdb

    pdb.set_trace()
    while True:
        print(obs[ObsProcessorTypes.axtree])
        action = get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    # release the environment
    env.close()


if __name__ == "__main__":
    main()

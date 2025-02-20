import gymnasium as gym
from computergym import make_env


def get_action(obs: dict):
    return 0


def main():
    env = make_env(
        "OpenEndedWebsite-v0", "browser", ["obs_processor_1", "obs_processor_2"]
    )

    obs, truncated, terminated = env.reset()
    while not terminated and not truncated:
        action = get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

import browsergym.core  # register the openended task as a gym environment
import gymnasium as gym
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html

env = gym.make(
    "browsergym/openended",
    task_kwargs={"start_url": "https://www.lawyersaathi.com/"},  # starting URL
    wait_for_user_message=True,  # wait for a user message after each agent message sent to the chat
    headless=False,  # run the browser in headless mode
)


def obs_preprocessor(obs: dict) -> dict:

    return {
        "chat_messages": obs["chat_messages"],
        "screenshot": obs["screenshot"],
        "goal_object": obs["goal_object"],
        "last_action": obs["last_action"],
        "last_action_error": obs["last_action_error"],
        "open_pages_urls": obs["open_pages_urls"],
        "open_pages_titles": obs["open_pages_titles"],
        "active_page_index": obs["active_page_index"],
        "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
        "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
    }


# run the environment <> agent loop until termination
obs, info = env.reset()
obs = obs_preprocessor(obs)

import pdb

pdb.set_trace()
while True:
    action = ...  # implement your agent here
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
# release the environment
env.close()

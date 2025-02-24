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
    score = 0
    n_tasks = 0
    for seed in range(1):
        for task_entrypoint in SERVICE_CATALOG_TASKS:
            task = task_entrypoint(seed=seed)
            goal, _ = task.setup_goal(None)

            print("Task:", task_entrypoint)
            print("Goal:", goal)

            env: OpenEndedWebsite = make_env(
                "lawyersaathi-v0",
                # "https://lawyersaathi.com",
                task.start_url,
                # "https://dev283325.service-now.com/now/nav/ui/classic/params/target/catalog_home.do%3Fsysparm_view%3Dcatalog_default",
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
            logger.info(
                f"Reward: {reward}, Stop: {stop}, Message: {message}, Info: {info}"
            )
            env.close()
            score += reward
            n_tasks += 1
    print(f"Total Score: {score}")
    print(f"Total Tasks: {n_tasks}")
    print(f"Average Score: {score / n_tasks}")


if __name__ == "__main__":
    main()

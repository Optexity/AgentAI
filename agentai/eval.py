import argparse
import os


def main(path: str):
    total = 0
    correct = 0
    for task in sorted(os.listdir(path)):
        task_path = os.path.join(path, task)
        for seed in sorted(os.listdir(task_path)):
            seed_path = os.path.join(task_path, seed)
            if os.path.isdir(seed_path):
                for file in os.listdir(seed_path):
                    if file == "agent.log":
                        with open(os.path.join(seed_path, file), "r") as f:
                            content = f.read()
                        total += 1
                        if "Final Reward: 1" in content:
                            correct += 1
                        else:
                            print(f"Incorrect: {task} {seed}")

    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct / total:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the browser gym tasks.")
    parser.add_argument("--log_path", type=str, default="./logs")
    args = parser.parse_args()
    try:
        main(args.log_path)
    except Exception as e:
        print(f"Error: {e}")

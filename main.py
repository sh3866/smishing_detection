from fire import Fire
import mmlu
# import pretraining
import trivia_qa
import phishing
# import earnings_qa
import torch
import numpy as np
import random


def main(task_name: str, **kwargs):
    torch.manual_seed(kwargs["seed"])
    np.random.seed(kwargs["seed"])
    random.seed(kwargs["seed"])
    torch.cuda.manual_seed(kwargs["seed"])

    print(f"Running task {task_name} with kwargs {kwargs}")

    task_map = dict(
        mmlu=mmlu.main,
        phishing=phishing.main,
        # pretrain=pretraining.main,
        trivia_qa=trivia_qa.main,
        # earnings_qa=earnings_qa.main,
    )

    if task_name in task_map.keys():
        task_fn = task_map.get(task_name)  # get value

        if task_fn is None:
            raise ValueError(
                f"{task_name}. Choose from {list(task_map.keys())}")
        score = task_fn(**kwargs)
        result = {task_name: score}

    results = {name: round(score * 100, 2) for name, score in result.items()}
    print(results)
    return results


if __name__ == '__main__':
    Fire(main)

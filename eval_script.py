from llm_code_eval.evaluator import evaluate, evaluate_qwen, evaluate_qwen_coder
import re
import json
from tqdm import tqdm
from typing import Dict
import argparse


INPUT_FILE = "data/humaneval/humaneval_python_grade_evalplus_ratio.json"

REF_FILE = "human-eval-v2-20210705.jsonl"

def load_humaneval(path: str) -> Dict[str, dict]:
    tasks = {}
    with open(path, "r") as f:
        for line in f:
            task = json.loads(line)
            tasks[task["task_id"]] = task
    return tasks



def is_solution_key(key):
    """
    Returns True if the key corresponds to a generated solution.
    e.g. "141", "20"
    """
    return re.fullmatch(r"\d+", key) is not None


def main():

    parser = argparse.ArgumentParser(description='Argparse Tutorial')
    parser.add_argument('--ref',          type=str,   default="False")
    args    = parser.parse_args()


    with open(INPUT_FILE, "r") as f:
        data = json.load(f)


    humaneval_tasks = load_humaneval(REF_FILE)

    results = []

    for task in tqdm(data):

        problem = task["intent"]
        task_id = task.get("task_id")
        
        ref = humaneval_tasks["HumanEval/"+task_id]["canonical_solution"]
        # print(ref)

        task_result = {
            "task_id": task_id,
            "evaluations": {}
        }

        # print(args.ref)

        for key, value in task.items():

            if not is_solution_key(key):
                continue

            generated_code = value

            try:
                if args.ref == "True":
                    score = evaluate_qwen(
                        problem=problem,
                        output=generated_code,
                        reference=ref,
                        task="code-gen",
                        aspect="functional correctness",
                        cot=False
                    )
                else:
                    score = evaluate_qwen(
                        problem=problem,
                        output=generated_code,
                        reference=None,
                        task="code-gen",
                        aspect="functional correctness",
                        cot=False
                    )


            except Exception as e:
                print(f"Evaluation failed for task {task_id} solution {key}: {e}")
                score = None

            print(score)

            task_result["evaluations"][key] = score

        results.append(task_result)

    if args.ref == "True":
        output_file = "data/humaneval/humaneval_llm_eval_qwen_c_ref.json"
    else:
        output_file = "data/humaneval/humaneval_llm_eval_qwen_c.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation finished.")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()


# score = evaluate_qwen_coder(problem="Given a list of integers, return the sum of all the integers.", 
#                 output="sum = 0\nfor i in range(len(list)):\n\tsum += list[i]\nreturn sum", 
#                 reference="sum = 0\nfor i in range(len(list)):\n\tsum += list[i]\nreturn sum", 
#                 task="code-gen", aspect="functional correctness")

# print(score)
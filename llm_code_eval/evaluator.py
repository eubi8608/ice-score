import openai
from collections import Counter
from .utils import TASK_PROMPTS
import requests


client = openai.OpenAI(base_url='http://127.0.0.1:5000/v1', api_key='') 


SERVER_URL = "http://127.0.0.1:5000/request"
# MODEL_NAME = "Qwen/Qwen3-Coder-Next"
MODEL_NAME = "Qwen/Qwen3.5-122B-A10B"
MODEL_NAME = "qwen-3-next"


def get_gpt_answer(raw_content, aspect):
    """
    Extracts the GPT answer from the raw content.
    
    Args:
        raw_content (str): The raw content from GPT response.
        aspect (str): The evaluation aspect.

    Returns:
        int: The extracted answer as an integer.
    """
    try:
        # print(raw_content)
        return int(raw_content)
    except ValueError:
        try:
            return process_raw_content(raw_content, aspect)
        except:
            return 0
        

def get_qwen_answer(raw_content, aspect):
    """
    Extracts the GPT answer from the raw content.
    
    Args:
        raw_content (str): The raw content from GPT response.
        aspect (str): The evaluation aspect.

    Returns:
        int: The extracted answer as an integer.
    """
    try:
        # print(raw_content['response'])
        return float(raw_content['response'])
    except ValueError:
        try:
            return process_raw_content(raw_content['response'], aspect)
        except:
            return 0


def process_raw_content(content, aspect):
    """
    Processes the raw content to extract the answer.
    
    Args:
        content (str): The raw content from GPT response.
        aspect (str): The evaluation aspect.

    Returns:
        int: The extracted answer as an integer.
    """
    # Normalize content: lowercase, remove parentheses, and split into lines
    splits = content.lower().replace("(", "").replace(")", "").split("\n")
    
    # Extract lines containing "score", remove dots, and replace "out of" and "/4"
    ls = [
        ll.strip(".").replace("out of ", "/").replace("/4", "")
        for l in splits
        # for ll in l.lstrip('0123456789. ').split(". ")
        for ll in l.split(". ")
        if any(item in ll for item in ["score"] + aspect.split())
    ]
    
    # Extract all numeric characters in each line and store them in a list
    ans = [ll for l in ls for ll in l.split() if ll.isnumeric()]
    
    # If there are multiple answers, return the most common one
    if len(set(ans)) != 1 and len(ans) > 1:
        return int(Counter(ans).most_common(1)[0][0])
    
    # Handle special cases where there are no answers or multiple non-numeric answers
    if len(set(ans)) != 1:
        if "N/A" in content:
            return 0
            
    # Return the single numeric answer
    return float(ans[0])

def evaluate(problem, output, reference=None, task="code-gen", aspect="usefulness", model="gpt-3.5-turbo", cot=False):
    """
    Evaluates the given problem and output using GPT.
    
    Args:
        problem (str): The problem statement.
        output (str): The output of the problem.
        reference (str, optional): The reference solution. Defaults to None.
        task (str, optional): The task type. Defaults to "code-gen".
        aspect (str, optional): The evaluation aspect. Defaults to "usefulness".
        model (str, optional): The GPT model to use. Defaults to "gpt-3.5-turbo".
        cot (bool, optional): Indicates if step-by-step evaluation is required. Defaults to False.

    Returns:
        Union[int, Tuple[int, str]]: The evaluation score or a tuple containing the score and step-by-step evaluation.
    """
    if reference:
        prompts = TASK_PROMPTS[task][aspect]["reference-enhanced"]
    else:
        prompts = TASK_PROMPTS[task][aspect]["reference-free"]
        
    if cot:
        prompts = prompts.replace(" (scores ONLY):", ":\nStep-by-step Evaluation:")
    
    if reference:
        prompts = prompts.replace("{{PROBLEM}}", problem).replace("{{OUTPUT}}", output).replace("{{REFERENCE}}", reference)
    else:
        prompts = prompts.replace("{{PROBLEM}}", problem).replace("{{OUTPUT}}", output)
        
    response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": prompts}],
            temperature=0,
    )

    raw_output = response["choices"][0]["message"]["content"]

    if cot:
        return get_gpt_answer(raw_output, aspect), raw_output
    else:
        return get_gpt_answer(raw_output, aspect)



def evaluate_qwen(problem, output, reference=None, task="code-gen", aspect="usefulness", model=MODEL_NAME, cot=False):
    """
    Evaluates the given problem and output using Qwen model via local server.
    """

    if reference:
        prompts = TASK_PROMPTS[task][aspect]["reference-enhanced"]
    else:
        prompts = TASK_PROMPTS[task][aspect]["reference-free"]

    if cot:
        prompts = prompts.replace(" (scores ONLY):", ":\nStep-by-step Evaluation:")

    if reference:
        prompts = prompts.replace("{{PROBLEM}}", problem)\
                         .replace("{{OUTPUT}}", output)\
                         .replace("{{REFERENCE}}", reference)
    else:
        prompts = prompts.replace("{{PROBLEM}}", problem)\
                         .replace("{{OUTPUT}}", output)

    payload = {
        "model": model,
        "system_msg": "You are a software engineering expert.",
        "prompt": prompts,
        "temperature": 0
    }

    response = requests.post(SERVER_URL, json=payload)
    response.raise_for_status()

    result = response.json()

    # 서버 응답 형식에 따라 조정
    if "choices" in result:
        raw_output = result["choices"][0]["message"]["content"]
    elif "text" in result:
        raw_output = result["text"]
    else:
        raw_output = result

    if cot:
        return get_qwen_answer(raw_output, aspect), raw_output
    else:
        return get_qwen_answer(raw_output, aspect)
    

def evaluate_qwen_coder(problem, output, reference=None, task="code-gen", aspect="usefulness", model=MODEL_NAME, cot=False):
    """
    Evaluates the given problem and output using GPT.
    
    Args:
        problem (str): The problem statement.
        output (str): The output of the problem.
        reference (str, optional): The reference solution. Defaults to None.
        task (str, optional): The task type. Defaults to "code-gen".
        aspect (str, optional): The evaluation aspect. Defaults to "usefulness".
        model (str, optional): The GPT model to use. Defaults to "gpt-3.5-turbo".
        cot (bool, optional): Indicates if step-by-step evaluation is required. Defaults to False.

    Returns:
        Union[int, Tuple[int, str]]: The evaluation score or a tuple containing the score and step-by-step evaluation.
    """
    if reference:
        prompts = TASK_PROMPTS[task][aspect]["reference-enhanced"]
    else:
        prompts = TASK_PROMPTS[task][aspect]["reference-free"]
        
    if cot:
        prompts = prompts.replace(" (scores ONLY):", ":\nStep-by-step Evaluation:")
    
    if reference:
        prompts = prompts.replace("{{PROBLEM}}", problem).replace("{{OUTPUT}}", output).replace("{{REFERENCE}}", reference)
    else:
        prompts = prompts.replace("{{PROBLEM}}", problem).replace("{{OUTPUT}}", output)
        
    response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", 
                       "content": "You are a software engineering expert.",
                       "role": "user",
                       "content": prompts + "\nPlease answer only the score without any other explanation.",}],
            temperature=0,
    )

    raw_output = response.choices[0].message.content
    # print(raw_output)

    if cot:
        return get_gpt_answer(raw_output, aspect), raw_output
    else:
        return get_gpt_answer(raw_output, aspect)


def evaluate_gpt5(problem, output, reference=None, task="code-gen", aspect="usefulness", model="gpt-5-nano", cot=False):
    """
    Evaluates the given problem and output using GPT.
    """

    if reference:
        prompts = TASK_PROMPTS[task][aspect]["reference-enhanced"]
    else:
        prompts = TASK_PROMPTS[task][aspect]["reference-free"]

    if cot:
        prompts = prompts.replace(" (scores ONLY):", ":\nStep-by-step Evaluation:")

    if reference:
        prompts = prompts.replace("{{PROBLEM}}", problem)\
                         .replace("{{OUTPUT}}", output)\
                         .replace("{{REFERENCE}}", reference)
    else:
        prompts = prompts.replace("{{PROBLEM}}", problem)\
                         .replace("{{OUTPUT}}", output)
    print(prompts)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": prompts 
                # + "\nPlease answer only the score without any other explanation."
            }
        ],
    )

    raw_output = response.output_text

    if cot:
        return get_gpt_answer(raw_output, aspect), raw_output
    else:
        return get_gpt_answer(raw_output, aspect)
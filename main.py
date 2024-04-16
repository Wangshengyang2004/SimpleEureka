import hydra
from omegaconf import DictConfig
from openai import OpenAI
from loguru import logger
import re
import subprocess
from pathlib import Path
import os
import datetime
import re
import shutil
from utils.extract_task_code import file_to_string

now = datetime.datetime.now()
logger.add(
    sink="se_{time}.log",
    rotation="1 day",
    retention="7 days",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

EUREKA_ROOT_DIR = os.getcwd()
RESULT_DIR = f"{EUREKA_ROOT_DIR}/results/{now}"

# Clean the results directory
shutil.rmtree(f"{EUREKA_ROOT_DIR}/results/{now}", ignore_errors=True)
os.makedirs(f"{EUREKA_ROOT_DIR}/results/{now}", exist_ok=True)

def clean_response(text):
    # Remove import statements
    text = re.sub(r'^\s*import .*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*from .* import .*$', '', text, flags=re.MULTILINE)

    # Remove all function definitions except magic methods like __init__, __str__, etc.
    text = re.sub(r'^\s*def (?!(?:__\w+__)\s*\().*$', '', text, flags=re.MULTILINE)

    # Find and remove everything after the last set of triple backticks
    last_backticks = text.rfind('```')
    if last_backticks != -1:
        text = text[:last_backticks]

    # Remove all triple backticks
    text = text.replace('```', '')

    # Clean up extra newlines
    text = re.sub(r'\n\s*\n', '\n', text)  # Reduce multiple newlines to single ones

    return text.strip()

# Define a type alias for the config object
@hydra.main(config_path="./", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    ISAAC_ROOT_DIR = cfg.gym.isaacsimpath
    PYTHON_PATH = cfg.gym.pythonpath
    prompt_dir = f"{EUREKA_ROOT_DIR}/prompts"

    logger.info(f"Current working directory: {EUREKA_ROOT_DIR}")
    # Load text from the prompt file
    initial_prompt = file_to_string(f"{prompt_dir}/initial.txt")
    task_description = file_to_string(f"{EUREKA_ROOT_DIR}/input/task_description.txt")
    crazyflie_examples = file_to_string(
        f"{EUREKA_ROOT_DIR}/input/crazyflie.py"
    ) + file_to_string(f"{EUREKA_ROOT_DIR}/input/crazyflie_human_diff.py")
    crazyflie_config = file_to_string(f"{EUREKA_ROOT_DIR}/input/crazyflie.yaml")
    tips = file_to_string(f"{prompt_dir}/tips.txt")
    must_do = file_to_string(f"{prompt_dir}/must_do.txt")
    final = "Add a sign for end of your code, when you finish the is_done part: #END\n"
    full_prompt = (
        initial_prompt
        + task_description
        + crazyflie_examples
        + crazyflie_config
        + tips
        + must_do
        + final
    )
    logger.info("Full Prompt: " + full_prompt)

    # Save the full prompt to a file under {EUREKA_ROOT_DIR}/results
    with open(f"{EUREKA_ROOT_DIR}/results/full_prompt.txt", "w") as f:
        f.write(full_prompt)

    client = OpenAI(
        api_key=cfg.api.key,
        base_url=cfg.api.url,
    )

    content = full_prompt
    messages = [
        {
            "role": "system",
            "content": "Use English to repond to the following prompts on RL code optimization task.",
        },
        {"role": "user", "content": f"{content}"},
    ]
    resp = []
    # Generate a response from the model using the full prompt until the #END sign
    for i in range(1, 3):
        completion = client.chat.completions.create(
            model=cfg.api.model,
            messages=messages,
            temperature=cfg.api.temperature,
            max_tokens=cfg.api.max_tokens,
        )
        if response := completion.choices[0].message.content:
            resp.append(response)
            logger.info(response)

            # Save the response to a file under {EUREKA_ROOT_DIR}/results
            with open(f"{EUREKA_ROOT_DIR}/results/{now}/response_{i}.txt", "w") as f:
                f.write(response)
            if re.search(r"#END", response):
                break
            else:
                messages.append(
                    {"role": "assistant", "content": f"{response}" + "continue\n"}
                )
    else:
        logger.info("Error: #END not found in the response or no response received.")

    # for i in range(len(resp)):
    #     resp[i] = clean_response(resp[i])
    logger.info(f"Cleaned responses:{resp}")
    pattern = r"def\s+(\w+)\s*\("
    functions = sum([re.findall(pattern, response) for response in resp], [])

    with open(f"{EUREKA_ROOT_DIR}/input/crazyflie.py", "r") as file:
        original_crazyflie_code = file.read()

    def remove_old_functions(code, function_names):
        for func_name in function_names:
            # Pattern to match the function and everything up until the start of the next function or end of the class
            pattern = r"^\s*def " + re.escape(func_name) + r".*?(?=\n\s*def |\Z)"
            # Replace the pattern with an empty string
            code = re.sub(pattern, "", code, flags=re.MULTILINE | re.DOTALL)
        return code

    updated_code = remove_old_functions(original_crazyflie_code, functions)

    # Append new code with correct indentation
    new_code_snippets = "\n    ".join(resp).replace("\n", "\n    ")
    updated_code += "\n    # New Code Starts Here\n    " + new_code_snippets

    def final_cleaner(code):
        # Remove the #END sign
        code = code.replace("#END", "")
        code = code.replace("```", "")
        # Remove the continue sign
        code = code.replace("```python", "")
        code = code.replace("python", "")
        return code
    
    updated_code = final_cleaner(updated_code)
    updated_file_path = f"{RESULT_DIR}/updated_crazyflie.py"
    # Rmove the updated crazyflie.py file if it exists
    if os.path.exists(updated_file_path):
        os.remove(updated_file_path)
    with open(updated_file_path, "w") as file:
        file.write(updated_code)
    
    logger.info(f"Updated crazyflie.py with new functions saved to {updated_file_path}")

    RUN = input("Do you want to run the updated code? (y/n): ")
    if RUN == "y":
        # Copy the updated crazyflie.py to the Isaac Sim directory
        shutil.copyfile(updated_file_path, f"{ISAAC_ROOT_DIR}/tasks/crazyflie.py")
        subprocess.call(
            args=f"cd {ISAAC_ROOT_DIR}; ~/.local/share/ov/pkg/isaac_sim-*/python.sh scripts/rlgames_train.py task=Crazyflie headless=true",
            shell=True,
            # use bin/bash to run the command
            executable="/bin/bash",
        )
    else:
        pass
    logger.info("Task complete, shutting down")
if __name__ == "__main__":
    main()

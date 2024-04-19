import hydra
from omegaconf import DictConfig
from openai import OpenAI
from loguru import logger
import re
import subprocess
import os
import datetime
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.extract_task_code import file_to_string
from utils.simple_eureka import remove_old_functions, final_cleaner, add_imports
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger.add(
    sink="./results/{time}/output.log",
    rotation="1 day",
    retention="7 days",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

EUREKA_ROOT_DIR = os.getcwd()
RESULT_DIR = f"{EUREKA_ROOT_DIR}/results/{now}"

# Clean the results directory
shutil.rmtree(RESULT_DIR, ignore_errors=True)
os.makedirs(RESULT_DIR, exist_ok=True)

def request_and_clean(cfg: DictConfig, full_prompt: str, original_crazyflie_code: str, task_id: int):
    client = OpenAI(api_key=cfg.api.key, base_url=cfg.api.url)
    messages = [
        {"role": "system", "content": "Use English to respond to the following prompts on RL code optimization task."},
        {"role": "user", "content": full_prompt}
    ]

    try:
        completion = client.chat.completions.create(
            model=cfg.api.model,
            messages=messages,
            temperature=cfg.api.temperature,
            max_tokens=cfg.api.max_tokens,
        )

        response = completion.choices[0].message.content if completion.choices else None
        if response:
            logger.info(response)
            with open(f"{EUREKA_ROOT_DIR}/results/{now}/response_{task_id}.txt", "w") as f:
                f.write(response)

            if "END" in response:
                logger.success("Successfully parsed response with #END")
            else:
                logger.error("Error: #END not found in the response or no response received.")
        else:
            logger.error("No response received from the model.")

    except Exception as e:
        logger.error(f"API call failed: {e}")
        response = None

    if response:
        resp = response.split("#END")[0]
        pattern = r"def\s+(\w+)\s*\("
        functions = sum([re.findall(pattern, response) for _ in resp], [])
        updated_code = remove_old_functions(original_crazyflie_code, functions)
        updated_code = add_imports(updated_code, resp)
        new_code_snippets = "\n    ".join(resp).replace("\n", "\n    ")
        updated_code += "\n    # New Code Starts Here\n    " + new_code_snippets
        updated_code = final_cleaner(updated_code)
        updated_file_path = f"{RESULT_DIR}/updated_crazyflie_{task_id}.py"

        if os.path.exists(updated_file_path):
            os.remove(updated_file_path)
        with open(updated_file_path, "w") as file:
            file.write(updated_code)
        logger.info(f"Updated crazyflie.py with new functions saved to {updated_file_path}")
        return updated_file_path, updated_code
    return None, None
                          

# Define a type alias for the config object
@hydra.main(config_path="./", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    #---------------------- SETUP ------------------#
    ISAAC_ROOT_DIR = cfg.gym.isaacsimpath
    PYTHON_PATH = cfg.gym.pythonpath
    TASK = cfg.gym.task
    SCRIPTS_DIR = cfg.gym.scriptpath
    HEADLESS = cfg.gym.headless
    MULTIRUN = cfg.gym.multirun
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

    with open(f"{EUREKA_ROOT_DIR}/input/crazyflie.py", "r") as file:
        original_crazyflie_code = file.read()

    # Save the full prompt to a file under {EUREKA_ROOT_DIR}/results
    with open(f"{EUREKA_ROOT_DIR}/results/full_prompt.txt", "w") as f:
        f.write(full_prompt)

    #---------------------- REQUEST AND CLEAN Async------------------#
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(request_and_clean, cfg, full_prompt, original_crazyflie_code, task_id) for task_id in range(cfg.generation.number)]
        results = [future.result() for future in as_completed(futures)]
        updated_files = [result for result in results if result is not None]
        file_paths = [file_path for file_path, _ in updated_files]
    # ---------------- RUN -----------------#
    RUN: bool = input("Do you want to run the updated code? (y/n): ")
    i: int = input("Enter the index of the file you want to run: ")
    if RUN == "y":
        # Copy the updated crazyflie.py to the Isaac Sim directory
        shutil.copyfile(file_paths[i], f"{ISAAC_ROOT_DIR}/tasks/crazyflie.py")

        if MULTIRUN:
            subprocess.call(
                args=f"cd {ISAAC_ROOT_DIR}; {PYTHON_PATH} {SCRIPTS_DIR} task={TASK} headless={HEADLESS} multirun={MULTIRUN}",
                shell=True,
                # use bin/bash to run the command
                executable="/bin/bash",
            )
        else:
            subprocess.call(
                args=f"cd {ISAAC_ROOT_DIR}; {PYTHON_PATH} {SCRIPTS_DIR} task={TASK} headless={HEADLESS}",
                shell=True,
                # use bin/bash to run the command
                executable="/bin/bash",
            )
    else:
        pass
    logger.info("Task complete, shutting down")

if __name__ == "__main__":
    main()

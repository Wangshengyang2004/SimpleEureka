"""
Future Work:
1. Clean up the code and add comments, make functions robust, learn from eureka
2. Folder created during running will follow these structures:
   - results
        - 05-01_12-30-45
            - logs
                - output.log
                - error.log
            - response
                - response_0.txt
                - response_1.txt
                - response_2.txt
                - response_3.txt
                - response_4.txt
            - updated_code
                - crazyflie_0
                    - updated_crazyflie_0.py
                    - videos
                        - video_0.mp4
                        - video_1.mp4
                        - video_2.mp4

                - crazyflie_1
                    - updated_crazyflie_1.py
                    - videos
                        - video_0.mp4
                        - video_1.mp4
                        - video_2.mp4

            - full_prompt.txt
"""

import hydra
from omegaconf import DictConfig
from openai import OpenAI
from loguru import logger
import re
import subprocess
import os
import datetime
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.extract_task_code import file_to_string
from utils.simple_eureka import remove_old_functions, final_cleaner, add_imports
import select
# from utils.file_utils import clean_empty_folders, remove_folders_with_output_log
import threading
import sys
from utils.system import check_system_encoding

platform = sys.platform
# Global lock for thread-safe file operations
file_lock = threading.Lock()
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = logger.opt(colors=True)
logger.add(
    sink=f"./results/{now}/output.log",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)

EUREKA_ROOT_DIR = os.getcwd()
# clean_empty_folders(f"{EUREKA_ROOT_DIR}/results")
# remove_folders_with_output_log(f"{EUREKA_ROOT_DIR}/results")
RESULT_DIR = f"{EUREKA_ROOT_DIR}/results/{now}"

# Clean the results directory
shutil.rmtree(RESULT_DIR, ignore_errors=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def request_and_clean(
    cfg: DictConfig, full_prompt: str, original_crazyflie_code: str, task_id: int
):
    client = OpenAI(api_key=cfg.api.key, base_url=cfg.api.url)
    messages = [
        {
            "role": "system",
            "content": "Use English to respond to the following prompts on RL code optimization task.",
        },
        {"role": "user", "content": full_prompt},
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
            try:
                with file_lock:
                    with open(
                        f"{EUREKA_ROOT_DIR}/results/{now}/response_{task_id}.txt", "w"
                    ) as f:
                        f.write(response)
            except Exception as e:
                logger.error(f"Failed to write to file: {e}")
                return None, None

            # Proceed with further processing only if a valid response was written successfully
            return process_response(cfg, response, original_crazyflie_code, task_id)
        else:
            logger.error("No response received from the model.")
            return None, None

    except Exception as e:
        logger.error(f"API call failed: {e}")
        return None, None


def process_response(cfg, response, original_crazyflie_code, task_id):
    if "END" in response:
        logger.success("Successfully parsed response with #END")
    else:
        logger.error("Error: #END not found in the response or no response received.")
        return None, None

    # Extract code snippets up to the '#END' marker
    resp = response.split("#END")[0]

    # Extract function names using regular expression
    pattern = r"def\s+(\w+)\s*\("
    functions = re.findall(pattern, resp)

    # Remove old functions and add new ones
    updated_code = remove_old_functions(original_crazyflie_code, functions)
    updated_code = add_imports(updated_code, resp)
    new_code_snippets = "\n    ".join(resp.split("\n"))
    updated_code += "\n    # New Code Starts Here\n    " + new_code_snippets
    updated_code = final_cleaner(updated_code)

    # Determine the updated file path
    updated_file_path = f"{RESULT_DIR}/updated_crazyflie_{task_id}.py"

    # Write the updated code to a file with thread-safe operations
    try:
        with file_lock:
            if os.path.exists(updated_file_path):
                os.remove(updated_file_path)
            with open(updated_file_path, "w") as file:
                file.write(updated_code)
        logger.info(
            f"Updated crazyflie.py with new functions saved to {updated_file_path}"
        )
        return updated_file_path, updated_code
    except Exception as e:
        logger.error(f"Failed to write updated code to file: {e}")
        return None, None


def single_train_on_windows(
    cfg, ISAAC_ROOT_DIR, PYTHON_PATH, SCRIPTS_DIR, TASK, HEADLESS
):
    driver = cfg.gym.isaacsimpath.split(":")[0]
    encoding = "utf-8" if check_system_encoding() else "gbk"
    p = subprocess.run(args=f"{driver}: & cd {ISAAC_ROOT_DIR} & {PYTHON_PATH} {SCRIPTS_DIR} task={TASK} headless={HEADLESS} ", shell=True,stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding=encoding)
    
    logger.info(p.stdout)
    if re.search("error running python", p.stdout):
        raise logger.error(
            "Running code in Omniverse is failed. There was an error running python"
        )


def multi_train_on_windows(
    cfg, ISAAC_ROOT_DIR, PYTHON_PATH, SCRIPTS_DIR, TASK, HEADLESS
):
    driver = cfg.gym.isaacsimpath.split(":")[0]
    encoding = "utf-8" if check_system_encoding() else "gbk"
    p = subprocess.run(
        args=f"{driver}: & cd {ISAAC_ROOT_DIR} & {PYTHON_PATH} -m torch.distributed.run --nnodes=1 --nproc_per_node=2 {SCRIPTS_DIR} task={TASK} headless={HEADLESS} multi_gpu=True",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding=encoding,
    )
    logger.info(p.stdout)
    # If There was an error running python contained in r
    if re.search("error running python", p.stdout):
        raise Exception(
            "Running code in Omniverse is failed. There was an error running python"
        )


def single_train_on_linux(ISAAC_ROOT_DIR, PYTHON_PATH, SCRIPTS_DIR, TASK, HEADLESS):
    p = subprocess.Popen(
        args=f"cd {ISAAC_ROOT_DIR}; {PYTHON_PATH} {SCRIPTS_DIR} task={TASK} headless={HEADLESS}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8" if check_system_encoding() else "gbk",
        # use bin/bash to run the command
        # executable="/bin/bash",
    )
    r = p.communicate()[0]
    logger.info(r)

    # If There was an error running python contained in r
    if re.search("error running python", r):
        raise Exception(
            "Running code in Omniverse is failed. There was an error running python"
        )


def multi_train_on_linux(ISAAC_ROOT_DIR, PYTHON_PATH, SCRIPTS_DIR, TASK, HEADLESS):
    p = subprocess.Popen(
        args=f"cd {ISAAC_ROOT_DIR}; {PYTHON_PATH} -m torch.distributed.run --nnodes=1 --nproc_per_node=2 {SCRIPTS_DIR} task={TASK} headless={HEADLESS} multi_gpu=True",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8" if check_system_encoding() else "gbk",
        # use bin/bash to run the command
        # executable="/bin/bash",
    )
    r = p.communicate()[0]
    logger.info(r)

    # If There was an error running python contained in r
    if re.search("error running python", r):
        raise Exception(
            "Running code in Omniverse is failed. There was an error running python"
        )


# Define a type alias for the config object
@logger.catch
@hydra.main(config_path="./config", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    # ---------------------- SETUP ------------------#
    logger.info("Starting Eureka...")
    logger.info(f"Config: {cfg}")
    ISAAC_ROOT_DIR = cfg.gym.isaacsimpath
    PYTHON_PATH = cfg.gym.pythonpath
    TASK = cfg.gym.task
    TASK_PATH = cfg.output.overwrite
    SCRIPTS_DIR = cfg.gym.scriptpath
    HEADLESS = cfg.gym.headless
    MULTIGPU = cfg.gym.multigpu
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

    # ---------------------- REQUEST AND CLEAN Async------------------#
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                request_and_clean, cfg, full_prompt, original_crazyflie_code, task_id
            )
            for task_id in range(cfg.generation.number)
        ]
        results = [future.result() for future in as_completed(futures)]
        updated_files = [result for result in results if result is not None]
        file_paths = sorted([file_path for file_path, _ in updated_files if file_path is not None])

    # ---------------- RUN -----------------#
    RUN: bool = input("Do you want to run the updated code? (y/n): ")

    if RUN == "y":
        i: int = int(
            input(
                "Enter the index of the file you want to run, from 0 to n, type -1 for running all one-by-one: "
            )
        )
        if i == -1:
            for n in range(len(file_paths)):
                # Copy the updated crazyflie.py to the Isaac Sim directory
                shutil.copyfile(file_paths[n], f"{TASK_PATH}")

                if MULTIGPU:
                    try:
                        if platform == "linux":
                            multi_train_on_linux(
                                ISAAC_ROOT_DIR, PYTHON_PATH, SCRIPTS_DIR, TASK, HEADLESS
                            )
                        else:
                            multi_train_on_windows(
                                cfg,
                                ISAAC_ROOT_DIR,
                                PYTHON_PATH,
                                SCRIPTS_DIR,
                                TASK,
                                HEADLESS,
                            )
                    except Exception as e:
                        logger.error(f"Failed to run the updated code: {e}")
                else:
                    try:
                        if platform == "linux":
                            single_train_on_linux(
                                ISAAC_ROOT_DIR, PYTHON_PATH, SCRIPTS_DIR, TASK, HEADLESS
                            )
                        else:
                            single_train_on_windows(
                                cfg,
                                ISAAC_ROOT_DIR,
                                PYTHON_PATH,
                                SCRIPTS_DIR,
                                TASK,
                                HEADLESS,
                            )
                    except Exception as e:
                        logger.error(f"Failed to run the updated code: {e}")

        else:
            # Copy the updated crazyflie.py to the Isaac Sim directory
            shutil.copyfile(file_paths[i], f"{TASK_PATH}")

            if MULTIGPU:
                try:
                    if platform == "linux":
                        multi_train_on_linux(
                            ISAAC_ROOT_DIR, PYTHON_PATH, SCRIPTS_DIR, TASK, HEADLESS
                        )
                    else:
                        multi_train_on_windows(
                            cfg,
                            ISAAC_ROOT_DIR,
                            PYTHON_PATH,
                            SCRIPTS_DIR,
                            TASK,
                            HEADLESS,
                        )
                except Exception as e:
                    logger.error(f"Failed to run the updated code: {e}")
            else:
                try:
                    if platform == "linux":
                        single_train_on_linux(
                            ISAAC_ROOT_DIR, PYTHON_PATH, SCRIPTS_DIR, TASK, HEADLESS
                        )
                    else:
                        single_train_on_windows(
                            cfg,
                            ISAAC_ROOT_DIR,
                            PYTHON_PATH,
                            SCRIPTS_DIR,
                            TASK,
                            HEADLESS,
                        )
                except Exception as e:
                    logger.error(f"Failed to run the updated code: {e}")
    else:
        logger.info("The code was not run. The user decided not to run the code.")

    logger.info("All Task complete, program shutting down")


if __name__ == "__main__":
    main()

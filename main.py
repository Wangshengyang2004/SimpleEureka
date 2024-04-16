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
    for i in range(1, 10):
        completion = client.chat.completions.create(
            model=cfg.api.model,
            messages=messages,
            temperature=cfg.api.temperature,
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

    # Parse the files to get functions we need: newly defined function, calculate_metric, is_done
    # Use the regex to extract the function names, names of the functions are defined in the prompt
    pattern = r"def\s*(\w+)\s*\("
    functions = []
    for i in range(len(resp)):
        functions.append(re.findall(pattern=pattern, string=resp[i]))
    logger.info(f"New Functions: {functions}")
    subprocess.call(
        args=f"cd {ISAAC_ROOT_DIR}; ~/.local/share/ov/pkg/isaac_sim-*/python.sh scripts/rlgames_train.py task=Crazyflie headless=true",
        shell=True,
        # use bin/bash to run the command
        executable="/bin/bash",
    )
    # subprocess.call(['lxterminal -e  python3 try.py'], cwd='/home/pi/try', shell=True)

if __name__ == "__main__":
    main()

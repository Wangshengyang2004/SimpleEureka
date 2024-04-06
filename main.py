import hydra
from omegaconf import DictConfig, OmegaConf
from typing import TypeAlias
from openai import OpenAI
import logging
import numpy as np 
import json
import logging 
import re
import subprocess
from pathlib import Path
import shutil
import time 
import os
import datetime

now = datetime.datetime.now()

from utils.extract_task_code import file_to_string
logging.basicConfig(level=logging.INFO)
EUREKA_ROOT_DIR = os.getcwd()
# Clean the results directory
shutil.rmtree(f'{EUREKA_ROOT_DIR}/results', ignore_errors=True)
os.makedirs(f'{EUREKA_ROOT_DIR}/results', exist_ok=True)

# Define a type alias for the config object
@hydra.main(config_path="./", config_name="config",version_base="1.1")
def main(cfg: DictConfig) -> None:
    
    ISAAC_ROOT_DIR = cfg.gym.isaacsimpath
    PYTHON_PATH = cfg.gym.pythonpath
    prompt_dir = f'{EUREKA_ROOT_DIR}/prompts'

    logging.info(f"Current working directory: {EUREKA_ROOT_DIR}")
    # Load text from the prompt file
    initial_prompt = file_to_string(f'{prompt_dir}/initial.txt')
    task_description = file_to_string(f'{EUREKA_ROOT_DIR}/input/task_description.txt')
    crazyflie_examples = file_to_string(f'{EUREKA_ROOT_DIR}/input/crazyflie.py') + file_to_string(f'{EUREKA_ROOT_DIR}/input/crazyflie_renzo.py')
    crazyflie_config = file_to_string(f"{EUREKA_ROOT_DIR}/input/crazyflie.yaml")
    extras = file_to_string(f'{EUREKA_ROOT_DIR}/input/extras.txt')
    tips = file_to_string(f'{prompt_dir}/tips.txt')
    must_do = file_to_string(f'{prompt_dir}/must_do.txt')
    punishment = file_to_string(f'{prompt_dir}/punish.txt')
    final = "Add a sign for end of your code, when you finish the is_done part: #END\n"
    full_prompt = initial_prompt + task_description + crazyflie_examples + crazyflie_config + extras + tips + must_do + punishment + final
    logging.info("Full Prompt: " + full_prompt)

    # Save the full prompt to a file under {EUREKA_ROOT_DIR}/results
    with open(f'{EUREKA_ROOT_DIR}/results/full_prompt.txt', 'w') as f:
        f.write(full_prompt)

    client = OpenAI(
    api_key=cfg.api.key,
    base_url=cfg.api.url,
    )
    
    content = full_prompt
    messages=[
                {"role": "system", "content": "Use English to repond to the following prompts on RL code optimization task."},
                {"role": "user", "content": f'{content}'}
            ]
    # Generate a response from the model using the full prompt until the #END sign
    for i in range(1, 10):
        completion = client.chat.completions.create(
            model = cfg.api.model,
            messages = messages,
            temperature=0.3,
        )
        response = completion.choices[0].message.content
        logging.info(response)
        # Save the response to a file under {EUREKA_ROOT_DIR}/results
        with open(f'{EUREKA_ROOT_DIR}/results/response_{i}.txt', 'w') as f:
            f.write(response)
        if "#END" in response:
            break
        else:
            messages.append({"role": "assistant", "content": f'{response}' + "continue\n"})
    logging.info("Response saved to file")

if __name__ == "__main__":
    main()
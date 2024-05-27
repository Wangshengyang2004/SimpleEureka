import time
import hydra
from omegaconf import DictConfig
from loguru import logger
import re
import subprocess
import os
import datetime
import shutil
import openai
from utils.extract_task_code import file_to_string
from utils.simple_eureka import process_response
import json
import sys
from utils.system import (
    check_system_encoding,
    clean_folder,
    copy_folder_sub,
)
from utils.misc import block_until_training
from pathlib import Path
from utils.agent import Agent
from utils.tensorboard_parser import tensorboard_parser
platform = sys.platform
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


# Define a type alias for the config object
@logger.catch
@hydra.main(config_path="./config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ---------------------- SETUP ------------------#
    logger.info(
        f"Starting Eureka with following config: {cfg} \n Workspace:{Path.cwd()} \n Project Root: {EUREKA_ROOT_DIR} \n"
    )
    ISAAC_ROOT_DIR = cfg.gym.omniisaacsimpathenv
    PYTHON_PATH = cfg.gym.pythonpath
    TASK = cfg.gym.task
    TASK_PATH = cfg.output.overwrite
    SCRIPTS_DIR = cfg.gym.scriptpath
    HEADLESS = cfg.gym.headless
    ENABLE_RECORDING = cfg.gym.enable_recording
    
    MULTIGPU = cfg.gym.multigpu
    prompt_dir = f"{EUREKA_ROOT_DIR}/prompts"
    logger.debug(f"Using LLM: {cfg.api.model} with API Key: {cfg.api.key}")
    code_feedback = file_to_string(f"{prompt_dir}/code_feedback.txt")
    task_obs_code_string = file_to_string(f"{EUREKA_ROOT_DIR}/input/{TASK}.py")
    policy_feedback = file_to_string(f"{prompt_dir}/policy_feedback.txt")
    execution_error_feedback = file_to_string(
        f"{prompt_dir}/execution_error_feedback.txt"
    )
    code_output_tip = file_to_string(f"{prompt_dir}/actor/code_output_tip.txt")
    DUMMY_FAILURE = -10000.0
    max_successes = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_reward_code_path = None
    clean_folder(f"{ISAAC_ROOT_DIR}/runs/{TASK}")
    
    # ---------------------- MESSAGE Assemble------------------#
    # Experiment can be done here by changing the prompt, must follow the message format
    actor_prompt = Agent(cfg=cfg)
    messages = actor_prompt.message()
    
    # ---------------------- Save Prompt ------------------#
    full_prompt = messages[0]["content"] + messages[1]["content"]
    logger.info("Full Prompt: " + full_prompt)
    # Save the full prompt to a file under {EUREKA_ROOT_DIR}/results
    with open(f"{RESULT_DIR}/full_prompt.txt", "w") as f:
        f.write(full_prompt)

    # ---------------------- Evolution ------------------#
    for iter in range(cfg.generation.epochs):
        BASE_DIR = f"{RESULT_DIR}/iter{iter}"
        # Make sub directories: code, reponses, tensorboard, and videos
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(f"{BASE_DIR}/{iter}", exist_ok=True)

        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size: int = cfg.generation.chunk_size
        logger.info(
            f"Iteration {iter}: Generating {cfg.generation.sample} samples with {cfg.api.model}"
        )
        client = openai.OpenAI(api_key=cfg.api.key, base_url=cfg.api.url)
        while True:
            if total_samples >= cfg.generation.sample:
                break
            for attempt in range(3):
                try:
                    response_cur = client.chat.completions.create(
                        model=cfg.api.model,
                        messages=messages,
                        temperature=cfg.api.temperature,
                        max_tokens=cfg.api.max_tokens,
                        n=chunk_size,
                    )
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logger.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logger.info("Code terminated due to too many failed attempts!")
                exit()

            responses.extend(response_cur.choices)
            prompt_tokens = response_cur.usage.prompt_tokens
            total_completion_token += response_cur.usage.completion_tokens
            total_token += response_cur.usage.total_tokens

        if cfg.generation.sample == 1:
            logger.info(
                f"Iteration {iter}: GPT Output:\n "
                + responses[0].message.content
                + "\n"
            )

        # logger Token Information
        logger.info(
            f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}"
        )

        code_runs = []
        rl_runs = []

        for response_id in range(cfg.generation.sample):
            # Clean up the omniverse isaac sim environment's output directory
            clean_folder(f"{ISAAC_ROOT_DIR}/runs/{TASK}")
            os.makedirs(f"{BASE_DIR}/{response_id}", exist_ok=True)
            os.makedirs(f"{BASE_DIR}/{response_id}/checkpoint", exist_ok=True)
            response_cur = responses[response_id].message.content
            # Save the response to a file
            with open(
                f"{BASE_DIR}/{response_id}/response.txt", "w", encoding="utf-8"
            ) as f:
                f.write(response_cur)
            logger.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            patterns = [
                r"```python(.*?)```",
                r"```(.*?)```",
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ]
            for pattern in patterns:
                code_string = re.search(pattern, response_cur, re.DOTALL)
                if code_string is not None:
                    code_string = code_string.group(1).strip()
                    break
            code_string = response_cur if not code_string else code_string

            # Remove unnecessary imports
            lines = code_string.split("\n")
            lines = [" " * 4 + line for line in lines]
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    code_string = "\n".join(lines[i:])
                    break

            code_runs.append(code_string)

            # Implement our code here, skip for now
            code_string, reward_only_string = process_response(
                code_string, task_obs_code_string
            )
            # Save the new environment code when the output contains valid code string!
            output_file = f"{BASE_DIR}/{response_id}/env_iter{iter}_response{response_id}.py"  # env_iter{iter}_response{response_id}.py
            try:
                with open(output_file, "w") as file:
                    file.writelines(code_string + "\n")
            except TypeError as e:
                logger.error(f"Error writing to file: {e}")
                continue

            with open(
                f"{BASE_DIR}/{response_id}/env_iter{iter}_response{response_id}_rewardonly.py",
                "w",
            ) as file:
                file.writelines(reward_only_string + "\n")

            shutil.copyfile(output_file, f"{TASK_PATH}")
            std_path = f"{BASE_DIR}/{response_id}/env_iter{iter}_response{response_id}_train.log"
            with open(std_path, "w") as f:
                encoding = "utf-8" if check_system_encoding() else "gbk"

                if platform == "win32":
                    driver = cfg.gym.omniisaacsimpathenv.split(":")[0]
                    if MULTIGPU:
                        if not HEADLESS:
                            logger.warning("Multi-GPU training is only supported in headless mode! Please set headless=False in the config. Making headless=True for now.")
                            HEADLESS = True
                        command = f"{driver}: & cd {ISAAC_ROOT_DIR} & {PYTHON_PATH} -m torch.distributed.run --nnodes=1 --nproc_per_node=2 {SCRIPTS_DIR} task={TASK} headless={HEADLESS} multi_gpu={MULTIGPU} "
                    else:
                        command = f"{driver}: & cd {ISAAC_ROOT_DIR} & {PYTHON_PATH} {SCRIPTS_DIR} task={TASK} headless={HEADLESS} "
                elif platform == "linux":
                    if MULTIGPU:
                        if not HEADLESS:
                            logger.warning("Multi-GPU training is only supported in headless mode! Please set headless=False in the config. Making headless=True for now.")
                            HEADLESS = True
                        command = f"cd {ISAAC_ROOT_DIR} && {PYTHON_PATH} -m torch.distributed.run --nnodes=1 --nproc_per_node=2 {SCRIPTS_DIR} task={TASK} headless={HEADLESS} multi_gpu={MULTIGPU} "
                    else:
                        command = f"cd {ISAAC_ROOT_DIR} && {PYTHON_PATH} {SCRIPTS_DIR} task={TASK} headless={HEADLESS} "
                else:
                    logger.error("Unsupported platform!")
                    exit()

                if ENABLE_RECORDING and not MULTIGPU:
                    os.makedirs(f"{BASE_DIR}/{response_id}/videos/", exist_ok=True)
                    command += f"enable_recording=True recording_dir={BASE_DIR}/{response_id}/videos/"
                else:
                    logger.info("Recording is disabled! Either enable recording or use multi-gpu training!")

                logger.info(f"Command: {command}")
                process = subprocess.Popen(
                    command, shell=True, stdout=f, stderr=f, encoding=encoding
                )
            block_until_training(
                std_path,
                success_keyword=cfg.env.success_keyword,
                failure_keyword=cfg.env.failure_keyword,
                log_status=True,
                iter_num=iter,
                response_id=response_id,
            )
            copy_folder_sub(
                f"{ISAAC_ROOT_DIR}/runs/{TASK}", f"{BASE_DIR}/{response_id}/"
            )
            rl_runs.append(process)

        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            rl_run.communicate()
            # Determine the success of the code run
            success, reward = process_rl_run_results(rl_run)
            max_successes.append(success)
            if success > max_success_overall:
                max_success_overall = success
                max_reward_code_path = code_run
            
            # Store execution results and their rewards
            with open(f"{BASE_DIR}/{response_id}/results.json", 'w') as f:
                results_data = {
                    "success": success,
                    "reward": reward,
                    "code": code_run
                }
                json.dump(results_data, f, indent=4)

            logger.info(f"Iteration {iter}: Code Run {response_id} - Success: {success}, Reward: {reward}")

        # After processing all responses for this iteration
        if max_success_overall == DUMMY_FAILURE:
            logger.error("No successful runs in this iteration.")
        else:
            logger.success(f"Iteration {iter}: Best Code Path - {max_reward_code_path}")

        # Cleanup for the next iteration
        best_code_paths.append(max_reward_code_path)
        max_success_overall = DUMMY_FAILURE  # Reset for the next iteration

        # Optional: Analyze TensorBoard data to extract performance metrics
        tensorboard_data = tensorboard_parser(f"{ISAAC_ROOT_DIR}/runs/{TASK}")
        execute_rates.append(tensorboard_data['execution_rate'])

        # Summary after all iterations
        if iter == cfg.generation.epochs - 1:
            logger.info("Finalizing and cleaning up...")
            summarize_experiment_results(execute_rates, max_successes, best_code_paths)

# Function to process and determine success from RL run logs or outputs
def process_rl_run_results(rl_run):
    # Placeholder for actual result processing logic
    success = False
    reward = 0
    # Implementation needed to parse rl_run output or logs to determine success and calculate reward
    return success, reward

# Function to summarize and log the overall experiment results
def summarize_experiment_results(execute_rates, max_successes, best_code_paths):
    average_execution_rate = sum(execute_rates) / len(execute_rates)
    highest_success_rate = max(max_successes)
    logger.info(f"Average Execution Rate: {average_execution_rate}")
    logger.info(f"Highest Success Rate: {highest_success_rate}")
    logger.info("Best Performing Code Paths:")
    for path in best_code_paths:
        logger.info(path)





if __name__ == "__main__":
    main()

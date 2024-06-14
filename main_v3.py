import time
import hydra
from omegaconf import DictConfig
from loguru import logger
import re
import subprocess
import os
import datetime
import shutil
import numpy as np
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
from utils.misc import block_until_training, construct_run_log, filter_traceback
from pathlib import Path
from utils.agent import Agent
from utils.tensorboard_parser import tensorboard_parser

platform = sys.platform
# Exit if macOS is detected
if platform == "darwin":
    logger.error("macOS is not supported! Exiting...")
    exit()
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = logger.opt(colors=True)
EUREKA_ROOT_DIR = os.getcwd()
RESULT_DIR = f"{EUREKA_ROOT_DIR}/results/{now}"
shutil.rmtree(RESULT_DIR, ignore_errors=True)
os.makedirs(RESULT_DIR, exist_ok=True)
logger.add(
    sink=f"./results/{now}/output.log",
    enqueue=True,
    backtrace=True,
    diagnose=True,
)
config_name = "config_linux" if platform == "linux" else "config_windows"


@logger.catch
@hydra.main(config_path="./config", config_name=config_name, version_base="1.3")
def main(cfg: DictConfig) -> None:
    # ---------------------- SETUP ------------------#
    logger.info(
        f"Starting Eureka with following config: {cfg} \n Workspace:{Path.cwd()} \n Project Root: {EUREKA_ROOT_DIR} \n"
    )
    ISAAC_ROOT_DIR = cfg.gym.omniisaacsimpathenv
    PYTHON_PATH = cfg.gym.pythonpath
    TASK_NAME: str = cfg.gym.task
    TASK_NAME_PATH = cfg.output.overwrite
    SCRIPTS_DIR = cfg.gym.scriptpath
    HEADLESS = cfg.gym.headless
    ENABLE_RECORDING = cfg.gym.enable_recording

    MULTIGPU = cfg.gym.multigpu
    prompt_dir = f"{EUREKA_ROOT_DIR}/prompts"
    logger.debug(f"Using LLM: {cfg.api.model} with API Key: {cfg.api.key}")
    code_feedback = file_to_string(f"{prompt_dir}/code_feedback.txt")
    task_obs_code_string = file_to_string(
        f"{EUREKA_ROOT_DIR}/input/{TASK_NAME.lower()}.py"
    )
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
    try:
        clean_folder(f"{ISAAC_ROOT_DIR}/runs/{TASK_NAME}")
    except Exception as e:
        logger.warning(f"Error cleaning up Isaac Sim directory: {e}")
        os.makedirs(f"{ISAAC_ROOT_DIR}/runs/{TASK_NAME}", exist_ok=True)
        logger.info(f"Created new directory: {ISAAC_ROOT_DIR}/runs/{TASK_NAME}")

    # ---------------------- MESSAGE Assemble------------------#
    actor_prompt = Agent(cfg=cfg)
    messages = actor_prompt.message()
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
                        messages=messages, # type: ignore
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
            if response_cur.usage is None:
                continue
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
            clean_folder(f"{ISAAC_ROOT_DIR}/runs/{TASK_NAME}")
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
            code_string, reward_only_string = process_response(
                code_string, task_obs_code_string
            )
            if code_string is None or reward_only_string is None:
                logger.error(
                    f"Error processing response {response_id}, skipping to next response..."
                )
                continue
            # Save the new environment code when the output contains valid code string!
            output_file = (
                f"{BASE_DIR}/{response_id}/env_iter{iter}_response{response_id}.py"
            )
            try:
                with open(output_file, "w", encoding="utf-8") as file:
                    file.writelines(code_string + "\n")
            except TypeError as e:
                logger.error(f"Error writing to file: {e}")
                continue

            with open(
                f"{BASE_DIR}/{response_id}/env_iter{iter}_response{response_id}_rewardonly.py",
                "w",
                encoding="utf-8",
            ) as file:
                file.writelines(reward_only_string + "\n")

            shutil.copyfile(output_file, f"{TASK_NAME_PATH}")
            std_path = f"{BASE_DIR}/{response_id}/env_iter{iter}_response{response_id}_train.log"
            with open(std_path, "w") as f:
                encoding = "utf-8" if check_system_encoding() else "gbk"

                if platform == "win32":
                    driver = cfg.gym.omniisaacsimpathenv.split(":")[0]
                    if MULTIGPU:
                        command = f"{driver}: & cd {ISAAC_ROOT_DIR} & {PYTHON_PATH} -m torch.distributed.run --nnodes=1 --nproc_per_node=2 {SCRIPTS_DIR} task={TASK_NAME} headless={HEADLESS} multi_gpu={MULTIGPU} "
                    else:
                        command = f"{driver}: & cd {ISAAC_ROOT_DIR} & {PYTHON_PATH} {SCRIPTS_DIR} task={TASK_NAME} headless={HEADLESS} "
                elif platform == "linux":
                    if MULTIGPU:
                        command = f"cd {ISAAC_ROOT_DIR} && {PYTHON_PATH} -m torch.distributed.run --nnodes=1 --nproc_per_node=2 {SCRIPTS_DIR} task={TASK_NAME} headless={HEADLESS} multi_gpu={MULTIGPU} "
                    else:
                        command = f"cd {ISAAC_ROOT_DIR} && {PYTHON_PATH} {SCRIPTS_DIR} task={TASK_NAME} headless={HEADLESS} "
                else:
                    logger.error("Unsupported platform!")
                    exit()

                if ENABLE_RECORDING and not MULTIGPU:
                    command += "enable_recording=True"
                elif not ENABLE_RECORDING and MULTIGPU:
                    command += "enable_recording=False"
                else:
                    logger.warning(
                        "Recording is disabled! Either enable recording or use multi-gpu training!"
                    )

                # TODO: Add Wandb support
                # command += f"hydra.run.dir={BASE_DIR}/{response_id} checkpoint={BASE_DIR}/{response_id}/checkpoint/"
                # if cfg.use_wandb:
                #     command.append("--no-wandb")
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
                f"{ISAAC_ROOT_DIR}/runs/{TASK_NAME}", f"{BASE_DIR}/{response_id}/"
            )
            rl_runs.append(process)

        # Gather RL training results and construct reward reflection
        code_feedbacks = []
        contents = []
        successes = []
        code_paths = []

        exec_success = False

        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, rl_runs)):
            rl_run.communicate()
            rl_filepath = f"{BASE_DIR}/{response_id}/env_iter{iter}_response{response_id}_train.log"
            code_paths.append(
                f"{BASE_DIR}/{response_id}/env_iter{iter}_response{response_id}.py"
            )

            try:
                with open(rl_filepath, "r") as f:
                    stdout_str = f.read()
            except Exception as e:
                logger.error(f"Error reading RL output log: {e}")
                content = "Code Run cannot be executed due to response parsing error!"
                contents.append(content) 
                successes.append(DUMMY_FAILURE)
                continue
            
            content = ""
            traceback_msg = filter_traceback(stdout_str)
            if traceback_msg == '':
                # No Error
                exec_success = True
                # Parse Run log
                # run_log = construct_run_log(stdout_str)
                # Parse tensorboard log
                tensorboard_logpath = f"{BASE_DIR}/{response_id}/summaries"
                tb_parser = tensorboard_parser(
                    tensorboard_logpath,
                    save=True,
                    plot=False,
                    dir_path=f"{BASE_DIR}/{response_id}/plot",
                    name=f"run_{response_id}",
                )
                tb_parser.parse_and_plot(dpi=150)
                tb_df = tb_parser.parse()
                # Aggregate policy-related feedback using details from run_log
                policy_feedback = policy_feedback.format(
                    sample_size="20",
                    tb_df=tb_df.to_string(),
                )
                content += policy_feedback
                code_feedbacks.append(code_feedback)
                content += code_feedback
                successes.append(tb_df.loc[24,"Max"])
            else:
                content = execution_error_feedback.format(traceback_msg=traceback_msg)
                successes.append(DUMMY_FAILURE)
            
            content += code_output_tip
            contents.append(content)

        # Additional code to determine the best response and handle feedback for the next iteration
        if not exec_success and cfg.generation.sample != 1:
            execute_rates.append(0.0)
            max_successes.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logger.info("All code generation failed! Repeating this iteration.")
            continue

        best_sample_idx = np.argmax(np.array(successes))
        best_content = contents[best_sample_idx]
        max_success = successes[best_sample_idx]
        execute_rate = np.sum(np.array(successes) >= 0.0) / cfg.generation.sample

        if max_success > max_success_overall:
            max_success_overall = max_success
            max_reward_code_path = code_paths[best_sample_idx]

        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        best_code_paths.append(code_paths[best_sample_idx])

        logger.info(
            f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}"
        )
        logger.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")

        # Update messaging for next iteration
        if len(messages) == 2:
            messages.append(
                {
                    "role": "assistant",
                    "content": responses[best_sample_idx].message.content,
                }
            )
            messages.append({"role": "user", "content": best_content})
        else:
            assert len(messages) == 4
            messages[-2] = {
                "role": "assistant",
                "content": responses[best_sample_idx].message.content,
            }
            messages[-1] = {"role": "user", "content": best_content}

        # Save the updated messages to a JSON file
        with open(f"{BASE_DIR}/messages.json", "w") as file:
            json.dump(messages, file, indent=4)

        if max_reward_code_path is None:
            logger.error("All iterations of code generation failed, aborting...")
            exit()

        logger.info(
            f"Max Training Success {max_success_overall}, Best Reward Code Path: {max_reward_code_path}"
        )

        best_reward = file_to_string(max_reward_code_path)
        with open(output_file.replace(".py", ".txt"), "w") as file:
            file.writelines(best_reward + "\n")

    if cfg.evaluation:
        logger.info("Starting Evaluation...")
        try:
            result = subprocess.run(['python', 'test_checkpoint.py', 'run_all=True'], shell=True, capture_output=True, text=True, check=True)
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(e.stderr)
            exit()

if __name__ == "__main__":
    main() # type: ignore

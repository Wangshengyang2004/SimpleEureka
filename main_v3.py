import time
import hydra
from matplotlib import pyplot as plt
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
from utils.simple_eureka import process_response, task_description_optimizer
import json
import sys
from utils.system import (
    check_system_encoding,
    clean_folder,
    copy_folder,
    copy_folder_sub,
)
from utils.misc import *
from utils.extract_task_code import *
from pathlib import Path

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
    # Clean up the omniverse isaac sim environment's output directory
    clean_folder(f"{ISAAC_ROOT_DIR}/runs/{TASK}")
    # MULTIGPU = cfg.gym.multigpu
    prompt_dir = f"{EUREKA_ROOT_DIR}/prompts"
    logger.info(f"Using LLM: {cfg.api.model} with API Key: {cfg.api.key}")

    def parse_api_doc(cfg):
        """Return the API document of OmniIsaacSim and functions of Omniverse"""
        path = f"{EUREKA_ROOT_DIR}/input/api_doc.txt"
        
    def critic_agent(cfg):
        """
        Set up the critic agent for the Eureka system
        This agent will explain the task description in detail and
        guide the reward function engineer on performance,
        creativity, and robustness of the reward function"""
        name = cfg.critic_agent.name
        fellow_name = cfg.actor_agent.name
        # Load prompt for critic agent
        critic_prompt = file_to_string(f"{prompt_dir}/critic_prompt.txt")
        task_description = file_to_string(f"{EUREKA_ROOT_DIR}/input/task_description.txt")
        system_instruction = file_to_string(f"{prompt_dir}/system_instruction.txt")
        template = critic_prompt.format(
            name=name,
            task_description=task_description, 
            fellow_name=fellow_name
        )
        messages = [
            {
                "role": "system",
                "content": system_instruction
            },
            {
                "role": "user",
                "content": template
            }
        ]
        client = openai.OpenAI(api_key=cfg.api.key, base_url=cfg.api.url)
        response_cur = client.chat.completions.create(
                            model=cfg.api.model,
                            messages=messages,
                            temperature=cfg.api.temperature,
                            max_tokens=cfg.api.max_tokens,
                            n=1
                        )
        return response_cur.choices[0].message.content
    
    def actor_agent(cfg):
        """
        Set up the actor agent for the Eureka system
        This agent will generate the reward function for the task description
        provided by the critic agent"""
        name = cfg.actor_agent.name
        # Load prompt for actor agent
        actor_prompt = file_to_string(f"{prompt_dir}/actor_prompt.txt")
        template = actor_prompt.format(name=name)
        messages = [
            {
                "role": "system",
                "content": template
            },
            {
                "role": "user",
                "content": "Optimize the code for the given task description."
            }
        ]
        client = openai.OpenAI(api_key=cfg.api.key, base_url=cfg.api.url)
        response_cur = client.chat.completions.create(
                            model=cfg.api.model,
                            messages=messages,
                            temperature=cfg.api.temperature,
                            max_tokens=cfg.api.max_tokens,
                            n=1
                        )
        return response_cur.choices[0].message.content
    # Load text from the prompt file
    initial_system = file_to_string(f"{prompt_dir}/initial_system.txt")
    code_output_tip = file_to_string(f"{prompt_dir}/code_output_tip.txt")
    code_feedback = file_to_string(f"{prompt_dir}/code_feedback.txt")
    initial_user = file_to_string(f"{prompt_dir}/initial_user.txt")
    policy_feedback = file_to_string(f"{prompt_dir}/policy_feedback.txt")
    execution_error_feedback = file_to_string(
        f"{prompt_dir}/execution_error_feedback.txt"
    )
    task_description = file_to_string(f"{EUREKA_ROOT_DIR}/input/task_description.txt")
    human_code_diff = file_to_string(f"{EUREKA_ROOT_DIR}/input/crazyflie_human_diff.py")
    task_obs_code_string = file_to_string(f"{EUREKA_ROOT_DIR}/input/crazyflie.py")
    env_config = file_to_string(f"{EUREKA_ROOT_DIR}/input/crazyflie.yaml")
    must_do = file_to_string(f"{prompt_dir}/must_do.txt")
    final = "Add a sign for end of your code, when you finish the is_done part: #END\n"

    # New: task description optimization
    if cfg.task_analyzer.enable:
        task_description = task_description_optimizer(cfg, task_description)

    # Assemble the full prompt
    initial_user = initial_user.format(
        task_obs_code_string=task_obs_code_string,
        task_description=task_description,
        code_output_tip=code_output_tip,
        human_code_diff=human_code_diff,
        env_config=env_config,
        final=final,
        must_do=must_do,
    )
    messages = [
        {"role": "system", "content": initial_system},
        {"role": "user", "content": initial_user},
    ]
    full_prompt = initial_system + initial_user
    DUMMY_FAILURE = -10000.0
    max_successes = []
    max_successes_reward_correlation = []
    execute_rates = []
    best_code_paths = []
    max_success_overall = DUMMY_FAILURE
    max_success_reward_correlation_overall = DUMMY_FAILURE
    max_reward_code_path = None

    logger.info("Full Prompt: " + full_prompt)

    # Save the full prompt to a file under {EUREKA_ROOT_DIR}/results
    with open(f"{RESULT_DIR}/full_prompt.txt", "w") as f:
        f.write(full_prompt)

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
                + responses[0]["message"]["content"]
                + "\n"
            )

        # logger Token Information
        logger.info(
            f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}"
        )

        code_runs = []
        rl_runs = []
        # for reponse in responses:
        #     logger.info(reponse.message.content)
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
            output_file = f"{BASE_DIR}/{response_id}/{cfg.gym.task}_env_iter{iter}_response{response_id}.py"  # env_iter{iter}_response{response_id}.py
            try:
                with open(output_file, "w") as file:
                    file.writelines(code_string + "\n")
            except TypeError as e:
                logger.error(f"Error writing to file: {e}")
                continue

            with open(
                f"{BASE_DIR}/{response_id}/{cfg.gym.task}_env_iter{iter}_response{response_id}_rewardonly.py",
                "w",
            ) as file:
                file.writelines(reward_only_string + "\n")

            shutil.copyfile(output_file, f"{TASK_PATH}")
            std_path = f"{BASE_DIR}/{response_id}/env_iter{iter}_response{response_id}_train.log"
            with open(std_path, "w") as f:
                encoding = "utf-8" if check_system_encoding() else "gbk"

                if platform == "win32":
                    driver = cfg.gym.omniisaacsimpathenv.split(":")[0]
                    command = f"{driver}: & cd {ISAAC_ROOT_DIR} & {PYTHON_PATH} {SCRIPTS_DIR} task={TASK} headless={HEADLESS} "
                elif platform == "linux":
                    command = f"cd {ISAAC_ROOT_DIR} && {PYTHON_PATH} {SCRIPTS_DIR} task={TASK} headless={HEADLESS} "
                else:
                    logger.error("Unsupported platform!")
                    exit()

                if cfg.gym.enable_recording:
                    os.makedirs(f"{BASE_DIR}/{response_id}/videos/", exist_ok=True)
                    command += f"enable_recording=True recording_dir={BASE_DIR}/{response_id}/videos/"
                command += f"hydra.run.dir={BASE_DIR}/{response_id} checkpoint={BASE_DIR}/{response_id}/checkpoint/"
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
                f"{ISAAC_ROOT_DIR}/runs/{TASK}", f"{BASE_DIR}/{response_id}/"
            )
            rl_runs.append(process)

        # Gather RL training results and construct reward reflection
        code_feedbacks = []
        contents = []
        successes = []
        reward_correlations = []
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
                logger.error(f"Error reading RL output file: {e}")
                content = execution_error_feedback.format(
                    traceback_msg="Code Run cannot be executed due to function signature error! Please re-write an entirely new reward function!"
                )
                content += code_output_tip
                contents.append(content)
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                continue

            content = ""
            traceback_msg = filter_traceback(stdout_str)

            if traceback_msg == "":
                # If RL execution has no error, provide policy statistics feedback
                exec_success = True
                run_log = construct_run_log(stdout_str)

                train_iterations = np.array(run_log["iterations/"]).shape[0]
                epoch_freq = max(int(train_iterations // 10), 1)

                epochs_per_log = 10
                content += policy_feedback.format(
                    epoch_freq=epochs_per_log * epoch_freq
                )

                # Compute Correlation between Human-Engineered and GPT Rewards
                if "gt_reward" in run_log and "gpt_reward" in run_log:
                    gt_reward = np.array(run_log["gt_reward"])
                    gpt_reward = np.array(run_log["gpt_reward"])
                    reward_correlation = np.corrcoef(gt_reward, gpt_reward)[0, 1]
                    reward_correlations.append(reward_correlation)

                # Add reward components log to the feedback
                for metric in sorted(run_log.keys()):
                    if "/" not in metric:
                        metric_cur = [
                            "{:.2f}".format(x) for x in run_log[metric][::epoch_freq]
                        ]
                        metric_cur_max = max(run_log[metric])
                        metric_cur_mean = sum(run_log[metric]) / len(run_log[metric])
                        if "consecutive_successes" == metric:
                            successes.append(metric_cur_max)
                        metric_cur_min = min(run_log[metric])
                        if metric != "gt_reward" and metric != "gpt_reward":
                            if metric != "consecutive_successes":
                                metric_name = metric
                            else:
                                metric_name = "task score"
                            content += f"{metric_name}: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                        else:
                            # Provide ground-truth score when success rate not applicable
                            if "consecutive_successes" not in run_log:
                                content += f"ground-truth score: {metric_cur}, Max: {metric_cur_max:.2f}, Mean: {metric_cur_mean:.2f}, Min: {metric_cur_min:.2f} \n"
                code_feedbacks.append(code_feedback)
                content += code_feedback
            else:
                # Otherwise, provide execution traceback error feedback
                successes.append(DUMMY_FAILURE)
                reward_correlations.append(DUMMY_FAILURE)
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content)

        # Repeat the iteration if all code generation failed
        if not exec_success and cfg.generation.sample != 1:
            execute_rates.append(0.0)
            max_successes.append(DUMMY_FAILURE)
            max_successes_reward_correlation.append(DUMMY_FAILURE)
            best_code_paths.append(None)
            logger.info(
                "All code generation failed! Repeat this iteration from the current message checkpoint!"
            )
            continue

        # Select the best code sample based on the success rate
        best_sample_idx = np.argmax(np.array(successes))
        best_content = contents[best_sample_idx]

        max_success = successes[best_sample_idx]
        max_success_reward_correlation = reward_correlations[best_sample_idx]
        execute_rate = np.sum(np.array(successes) >= 0.0) / cfg.generation.sample

        # Update the best Eureka Output
        if max_success > max_success_overall:
            max_success_overall = max_success
            max_success_reward_correlation_overall = max_success_reward_correlation
            max_reward_code_path = code_paths[best_sample_idx]

        execute_rates.append(execute_rate)
        max_successes.append(max_success)
        max_successes_reward_correlation.append(max_success_reward_correlation)
        best_code_paths.append(code_paths[best_sample_idx])

        logger.info(
            f"Iteration {iter}: Max Success: {max_success}, Execute Rate: {execute_rate}, Max Success Reward Correlation: {max_success_reward_correlation}"
        )
        logger.info(f"Iteration {iter}: Best Generation ID: {best_sample_idx}")
        logger.info(
            f"Iteration {iter}: GPT Output Content:\n"
            + responses[best_sample_idx]["message"]["content"]
            + "\n"
        )
        logger.info(f"Iteration {iter}: User Content:\n" + best_content + "\n")

        if len(messages) == 2:
            messages += [
                {
                    "role": "assistant",
                    "content": responses[best_sample_idx]["message"]["content"],
                }
            ]
            messages += [{"role": "user", "content": best_content}]
        else:
            assert len(messages) == 4
            messages[-2] = {
                "role": "assistant",
                "content": responses[best_sample_idx]["message"]["content"],
            }
            messages[-1] = {"role": "user", "content": best_content}

        # Save dictionary as JSON file
        with open("messages.json", "w") as file:
            json.dump(messages, file, indent=4)

    if max_reward_code_path is None:
        logger.info("All iterations of code generation failed, aborting...")
        logger.info(
            "Please double check the output env_iter*_response*.txt files for repeating errors!"
        )
        exit()
    logger.info(
        f"Task: {cfg.gym.task}, Max Training Success {max_success_overall}, Correlation {max_success_reward_correlation_overall}, Best Reward Code Path: {max_reward_code_path}"
    )

    best_reward = file_to_string(max_reward_code_path)
    with open(output_file, "w") as file:
        file.writelines(best_reward + "\n")

    # Get run directory of best-performing policy
    with open(max_reward_code_path.replace(".py", ".txt"), "r") as file:
        lines = file.readlines()


if __name__ == "__main__":
    main()

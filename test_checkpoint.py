#TODO: Add a maximum testing time constraint for each test, seperate the Omniverse output and the main program output
import subprocess
import sys
import hydra
from omegaconf import DictConfig
from pathlib import Path
import os
from loguru import logger
import shutil
platform = sys.platform
if platform == "darwin":
    logger.error("macOS is not supported! Exiting...")
    exit()
config_name = "config_linux" if platform == "linux" else "config_windows"

@hydra.main(config_path="./config", config_name=config_name, version_base="1.3")
def main(cfg):
    """
    Run this code in root directory of the project, using the following command:
    python test_checkpoint.py result_dir: str | int ="latest"/2024-06-10_01-17-05 iter: int = 0 idx: int = 0
    """
    platform = sys.platform
    ISAAC_ROOT_DIR = cfg.gym.omniisaacsimpathenv
    PYTHON_PATH = cfg.gym.pythonpath
    TASK_NAME: str = cfg.gym.task
    TASK_NAME_PATH = cfg.output.overwrite
    SCRIPTS_DIR = cfg.gym.scriptpath
    RESULT_DIR = cfg.result_dir
    ITER = cfg.iter
    IDX = cfg.idx
    RUN_ALL = cfg.run_all
    CHECKPOINT_NAME = cfg.checkpoint_name
    RECORD_VIDEO = cfg.record_video
    test=True
    num_envs=64
    if platform == "win32":
            driver = cfg.gym.omniisaacsimpathenv.split(":")[0]
            command = f"{driver}: & cd {ISAAC_ROOT_DIR} & {PYTHON_PATH} {SCRIPTS_DIR} task={TASK_NAME} test={test} num_envs={num_envs}"
    elif platform == "linux":
        command = f"cd {ISAAC_ROOT_DIR} && {PYTHON_PATH} {SCRIPTS_DIR} task={TASK_NAME} test={test} num_envs={num_envs}"
    else:
        logger.error("Unsupported platform! Exiting...")
        exit()

    if RESULT_DIR == "latest":
            RESULT_DIR = sorted([x for x in os.listdir("results") if os.path.isdir(os.path.join("results", x))])[-1]
    RESULT_DIR = f"results/{RESULT_DIR}"
    
    def run_one(result_dir, iter, idx):
        if not isinstance(iter, int):
            # split the "iter0" to get the integer value
            iter = int(iter.split("iter")[1])
        # Check if nn directory exists and not empty
        if not os.path.exists(f"{result_dir}/iter{iter}/{idx}/nn") or not os.listdir(f"{result_dir}/iter{iter}/{idx}/nn"):
            raise FileNotFoundError(f"{result_dir}/iter{iter}/{idx}/nn")         
        CHECKPOINT_NAME = sorted([x for x in os.listdir(f"{result_dir}/iter{iter}/{idx}/nn") if os.path.isfile(os.path.join(f"{result_dir}/iter{iter}/{idx}/nn", x))], key=lambda x: os.path.getmtime(f"{result_dir}/iter{iter}/{idx}/nn/{x}"))[-1]
        code_path = f"{result_dir}/iter{iter}/{idx}/env_iter{iter}_response{idx}.py"
        # logger.info(f"Copying the crazyflie.py file to the original location: {TASK_NAME_PATH}")
        shutil.copyfile(code_path, TASK_NAME_PATH)
        checkpoint_path = f"{result_dir}/iter{iter}/{idx}/nn/{CHECKPOINT_NAME}"
        # logger.info(f"Checkpoint path: {checkpoint_path}")
        # logger.info("Checking if Checkpoint file exists...")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        logger.info(f"Executing command: {command}")
        process = subprocess.Popen(command, shell=True)
        process.wait()
        logger.info("Execution completed!")
    
    if RUN_ALL:
        # Run all the checkpoints under RESULT_DIR
        for iter in sorted([x for x in os.listdir(RESULT_DIR) if os.path.isdir(os.path.join(RESULT_DIR, x))]):
            for idx in sorted([x for x in os.listdir(f"{RESULT_DIR}/{iter}") if os.path.isdir(os.path.join(f"{RESULT_DIR}/{iter}", x))]):
                logger.info(f"Running for iter: {iter} and idx: {idx}")
                try:
                    run_one(RESULT_DIR, iter, idx)
                except FileNotFoundError as e:
                    logger.warning(f"Iter: {iter} Index: {idx} doesn't not contain a checkpoint: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error occurred: {e}")
                    continue
    else:
        logger.info(f"Running for iter: {ITER} and idx: {IDX}")
        try:
            run_one(RESULT_DIR, ITER, IDX)
        except FileNotFoundError as e:
            logger.warning(f"Iter: {ITER} Index: {IDX} doesn't not contain a checkpoint: {e}")
        except Exception as e:
            logger.error(f"Error occurred: {e}")

        

if __name__ == "__main__":
    main()
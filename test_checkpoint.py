# TODO: Add a maximum testing time constraint for each test, seperate the Omniverse output and the main program output
import subprocess
import sys
import hydra
from omegaconf import DictConfig
import os
from loguru import logger
import shutil
import pathlib
from subprocess import TimeoutExpired
from utils.simple_eureka import concat_videos

current_dir = pathlib.Path(__file__).parent.absolute()
platform = sys.platform
if platform == "darwin":
    logger.error("macOS is not supported! Exiting...")
    exit()
config_name = "config_linux" if platform == "linux" else "config_windows"


@hydra.main(config_path="./config", config_name=config_name, version_base="1.3")
def main(cfg: DictConfig):
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
    RECORD_BASE_DIR = cfg.record_base_dir
    CONCAT_VIDEO = cfg.concat_videos
    test = True
    num_envs = cfg.num_envs
    if platform == "win32":
        driver = cfg.gym.omniisaacsimpathenv.split(":")[0]
        command = f"{driver}: & cd {ISAAC_ROOT_DIR} & {PYTHON_PATH} {SCRIPTS_DIR} task={TASK_NAME} test={test} num_envs={num_envs} recording_length={cfg.recording_length} recording_interval={cfg.recording_interval} max_iterations={cfg.max_epochs}"
    elif platform == "linux":
        command = f"cd {ISAAC_ROOT_DIR} && {PYTHON_PATH} {SCRIPTS_DIR} task={TASK_NAME} test={test} num_envs={num_envs} recording_length={cfg.recording_length} recording_interval={cfg.recording_interval} max_iterations={cfg.max_epochs}"
    else:
        logger.error("Unsupported platform! Exiting...")
        exit()

    if RESULT_DIR == "latest":
        RESULT_DIR = sorted(
            [
                x
                for x in os.listdir("results")
                if os.path.isdir(os.path.join("results", x))
            ]
        )[-1]
    RESULT_DIR = f"results/{RESULT_DIR}"



    def run_one(result_dir, iter, idx, command, max_running_time=300):
        """
        Runs a single instance of the testing process with a maximum running time.
        
        :param result_dir: Directory containing the results.
        :param iter: Iteration number.
        :param idx: Index number.
        :param command: Command to run the simulation.
        :param max_running_time: Maximum running time in seconds.
        """
        if RECORD_VIDEO:
            RECORD_DIR = os.path.join(current_dir, f"{RECORD_BASE_DIR}/{result_dir}/iter{iter}/idx{idx}")
            command += f" enable_recording=True recording_dir={RECORD_DIR}"
        if not os.path.exists(f"{result_dir}/iter{iter}/{idx}/nn") or not os.listdir(f"{result_dir}/iter{iter}/{idx}/nn"):
            raise FileNotFoundError(f"{result_dir}/iter{iter}/{idx}/nn")
        
        CHECKPOINT_NAME = sorted(
            [x for x in os.listdir(f"{result_dir}/iter{iter}/{idx}/nn") if os.path.isfile(os.path.join(f"{result_dir}/iter{iter}/{idx}/nn", x))],
            key=lambda x: os.path.getmtime(f"{result_dir}/iter{iter}/{idx}/nn/{x}")
        )[-1]
        
        code_path = f"{result_dir}/iter{iter}/{idx}/env_iter{iter}_response{idx}.py"
        shutil.copyfile(code_path, TASK_NAME_PATH)
        checkpoint_path = f"{current_dir}/{result_dir}/iter{iter}/{idx}/nn/{CHECKPOINT_NAME}"
        command += f" checkpoint={checkpoint_path}"
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        
        logger.info(f"Executing command: {command}")
        try:
            process = subprocess.Popen(command, shell=True)
            process.wait(timeout=max_running_time)  # Set the timeout for the process
            logger.info("Execution completed successfully!")
        except TimeoutExpired:
            logger.warning(f"Process exceeded maximum time of {max_running_time} seconds and was terminated.")
            process.kill()  # Terminate the process if it runs longer than the timeout
            process.communicate()  # Clean up process resources
        
        if CONCAT_VIDEO and os.path.exists(RECORD_DIR):
            concat_videos(RECORD_DIR)

    if RUN_ALL:
        # Run all the checkpoints under RESULT_DIR
        for iter in sorted(
            [
                x
                for x in os.listdir(RESULT_DIR)
                if os.path.isdir(os.path.join(RESULT_DIR, x))
            ]
        ):
            for idx in sorted(
                [
                    x
                    for x in os.listdir(f"{RESULT_DIR}/{iter}")
                    if os.path.isdir(os.path.join(f"{RESULT_DIR}/{iter}", x))
                ]
            ):
                logger.info(f"Running for iter: {iter} and idx: {idx}")
                try:
                    run_one(RESULT_DIR, iter, idx, command)
                except FileNotFoundError as e:
                    logger.warning(
                        f"Iter: {iter} Index: {idx} doesn't not contain a checkpoint: {e}"
                    )
                    continue
                except Exception as e:
                    logger.error(f"Error occurred: {e}")
                    continue
    else:
        logger.info(f"Running for iter: {ITER} and idx: {IDX}")
        try:
            run_one(RESULT_DIR, ITER, IDX, command)
        except FileNotFoundError as e:
            logger.warning(
                f"Iter: {ITER} Index: {IDX} doesn't not contain a checkpoint: {e}"
            )
        except Exception as e:
            logger.error(f"Error occurred: {e}")


if __name__ == "__main__":
    main()

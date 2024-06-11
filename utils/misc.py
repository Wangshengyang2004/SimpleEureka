import re
import subprocess
import os
import json
import time
from loguru import logger

from utils.extract_task_code import file_to_string
from utils.exceptions import CODE_EXECUTION_VALUEERROR, CODE_EXECUTION_SYNTAXERROR
def set_freest_gpu():
    freest_gpu = get_freest_gpu()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(freest_gpu)

def get_freest_gpu():
    # Note: if this line breaks, you can provide an absolute path to gpustat instead
    sp = subprocess.Popen(['gpustat', '--json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_str, _ = sp.communicate()
    gpustats = json.loads(out_str.decode('utf-8'))
    freest_gpu = min(gpustats['gpus'], key=lambda x: x['memory.used'])
    return freest_gpu['index']

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_training(rl_filepath, success_keyword, failure_keyword, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the RL training has started before moving on
    start_time = time.time()
    last_update_time = start_time
    initial_mod_time = os.path.getmtime(rl_filepath)
    
    while True:
        rl_log = file_to_string(rl_filepath)
        if success_keyword in rl_log or failure_keyword in rl_log:
            if log_status and success_keyword in rl_log:
                logger.success(f"Iteration {iter_num}: Code Run {response_id} successfully training!")
            if log_status and failure_keyword in rl_log:
                logger.error(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            break
        
        if "Simulation App Shutting Down" in rl_log:
            logger.success(f"Code Indeed Run, but Unknown error make it stop earilier: Iteration {iter_num}: Code Run {response_id} simulation app shutting down!")
            break
        if "Max epochs reached" in rl_log:
            logger.warning(f"Iteration {iter_num}: Code Run {response_id} max epochs reached before any env terminated at least once!")
            logger.success("Training Done!")
            break

        current_mod_time = os.path.getmtime(rl_filepath)
        if current_mod_time > initial_mod_time:
            last_update_time = time.time()
            initial_mod_time = current_mod_time
        
        if time.time() - last_update_time > 180:
            logger.error(f"Iteration {iter_num}: Code Run {response_id} training timeout!")
            break
        
        time.sleep(10)  # Sleep for a short while to prevent excessive checking

def construct_run_log(stdout_str) -> dict:
    run_log = {}
    # lines = stdout_str.split('\n')
    # Detect Key errors
    run_log['Error'] = filter_traceback(stdout_str)
    if "SyntaxError: invalid syntax" in stdout_str:
        run_log['Success'] = False
        raise CODE_EXECUTION_SYNTAXERROR(run_log['Error'])
    elif "ValueError" in stdout_str:
        run_log['Success'] = False
        raise CODE_EXECUTION_VALUEERROR(run_log['Error'])
    else:
        run_log['KeyError'] = None
        run_log['Success'] = True
        # Use Regex to find saving checkpoints
        checkpoints_pattern = re.compile(r"last_(.*)")
        run_log['Checkpoints'] = checkpoints_pattern.findall(stdout_str)
        reward_pattern = re.compile(r"Reward: (.*)")
        run_log['Reward'] = reward_pattern.findall(stdout_str)
    return run_log

if __name__ == "__main__":
    try:
        stdout_str = file_to_string(r'H:\Omniverse\Library\isaac_sim-2023.1.1\SimpleEureka\results\2024-05-18_21-09-07\iter0\code\env_iter0_response7_train.log')
        run_log = construct_run_log(stdout_str)
        print(run_log)
    except CODE_EXECUTION_SYNTAXERROR as e:
        print(e)
    except CODE_EXECUTION_VALUEERROR as e:
        print(e)
    except Exception as e:
        print(e)
import os
import streamlit as st
import subprocess
import yaml
import sys
import shutil
import cv2
import torch
from PIL import Image
from loguru import logger
from collections import deque
from threading import Thread
from queue import Queue, Empty
from torchviz import make_dot

# Determine the platform-specific configuration file
platform = sys.platform
config_name = "config_linux" if platform == "linux" else "config_windows"

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

    def __repr__(self):
        return f"{self.__dict__}"

def load_config(yaml_path):
    with open(yaml_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return Config(config_dict)

# Load the configuration
yaml_path = os.path.join(".", "config", f"{config_name}.yaml")
cfg = load_config(yaml_path)

# Set paths from the configuration
ISAAC_ROOT_DIR = cfg.gym.omniisaacsimpathenv
PYTHON_PATH = cfg.gym.pythonpath
SCRIPTS_DIR = cfg.gym.scriptpath
TASK_PATH = cfg.output.overwrite  # Assuming TASK_PATH is defined in the YAML config
TASK = cfg.gym.task
BASE_DIR = "./tests"
os.makedirs(BASE_DIR, exist_ok=True)

success_keyword = cfg.env.success_keyword
failure_keyword = cfg.env.failure_keyword

# Function to read lines from stream
def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

# Function to run the command and display output in real-time
def run_and_display_stdout(success_keyword, failure_keyword, *cmd_with_args):
    result = subprocess.Popen(cmd_with_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    q_stdout = Queue()
    q_stderr = Queue()

    t_stdout = Thread(target=enqueue_output, args=(result.stdout, q_stdout))
    t_stderr = Thread(target=enqueue_output, args=(result.stderr, q_stderr))
    t_stdout.daemon = True
    t_stderr.daemon = True
    t_stdout.start()
    t_stderr.start()

    log_placeholder = st.empty()
    buffer = deque(maxlen=30)

    while True:
        line = None
        err_line = None

        try:
            line = q_stdout.get_nowait()
        except Empty:
            pass
        
        try:
            err_line = q_stderr.get_nowait()
        except Empty:
            pass

        if line:
            buffer.append(line.strip())
            log_placeholder.code("\n".join(buffer), language="bash")
        if err_line:
            buffer.append(err_line.strip())
            log_placeholder.code("\n".join(buffer), language="bash")

        # Check for success and failure keywords in both stdout and stderr
        if line and "MAX EPOCHS NUM!" in line or err_line and "MAX EPOCHS NUM!" in err_line:
            logger.success("Code Run Successfully!")
            st.success("Code Run Successfully!")
            break
        if line and failure_keyword in line or err_line and failure_keyword in err_line:
            logger.error("Code Run Failed!")
            st.error("Code Run Failed!")
            break

        if line and "Simulation App Shutting Down" in line or err_line and "Simulation App Shutting Down" in err_line:
            logger.success("Code Indeed Run, but Unknown error make it stop earlier: simulation app shutting down!")
            st.success("Code Indeed Run, but Unknown error make it stop earlier: simulation app shutting down!")
            break
        if line and "Max epochs reached" in line or err_line and "Max epochs reached" in err_line:
            logger.warning("Code Run max epochs reached before any env terminated at least once!")
            logger.success("Training Done!")
            st.warning("Code Run max epochs reached before any env terminated at least once!")
            break
        
        if result.poll() is not None:
            break

def test_with_this_file(file_path, headless=True, enable_recording=False, multigpu=False):
    parts = file_path.split(os.sep)
    response_id = parts[-2]
    iter_num = parts[-3].split("iter")[-1]
    date = parts[1]
    target_dir = os.path.join(BASE_DIR, f"{date}_env_iter{iter_num}_response{response_id}_train")
    os.makedirs(target_dir, exist_ok=True)
    std_path = os.path.join(target_dir, "omniverse.log")

    shutil.copyfile(file_path, TASK_PATH)  # Copy the file to TASK_PATH
    st.toast(f"{file_path} copied to {TASK_PATH}")
    headless_flag = "headless=True" if headless else "headless=False"

    if platform == "win32":
        if multigpu:
            command = ["cmd", "/c", f"cd /d {ISAAC_ROOT_DIR} && {PYTHON_PATH} -u -m torch.distributed.run --nnodes=1 --nproc_per_node=2  {SCRIPTS_DIR} task={TASK} {headless_flag} multi_gpu={multigpu} "]
        else:
            command = ["cmd", "/c", f"cd /d {ISAAC_ROOT_DIR} && {PYTHON_PATH} -u {SCRIPTS_DIR} task={TASK} {headless_flag}"]
    elif platform == "linux":
        if multigpu:
            command = ["bash", "-c", f"cd {ISAAC_ROOT_DIR} && {PYTHON_PATH} -u -m torch.distributed.run --nnodes=1 --nproc_per_node=2  {SCRIPTS_DIR} task={TASK} {headless_flag} multi_gpu={multigpu} "]
        else:
            command = ["bash", "-c", f"cd {ISAAC_ROOT_DIR} && {PYTHON_PATH} -u {SCRIPTS_DIR} task={TASK} {headless_flag}"]

    else:
        logger.error("Unsupported platform!")
        exit()

    st.info(f"Command: {' '.join(command)}")
    run_and_display_stdout("Training done", "Error", *command)

def display_file_content(file_path):
    _, ext = os.path.splitext(file_path)
    if ext in [".txt", ".py", ".log"]:
        with open(file_path, "r") as file:
            content = file.read()
            st.code(content, language=ext[1:])
    elif ext in [".png", ".jpg", ".jpeg", ".gif"]:
        image = Image.open(file_path)
        st.image(image, caption=file_path)
    elif ext in [".mp4", ".avi", ".mov"]:
        video_file = open(file_path, "rb")
        video_bytes = video_file.read()
        st.video(video_bytes)
    elif ext == ".pth":
        model = torch.load(file_path)
        if isinstance(model, dict) and 'model_state_dict' in model:
            st.write("Model state_dict keys:")
            st.write(list(model['model_state_dict'].keys()))
        else:
            st.write("Model keys:")
            st.write(list(model.keys()))
        
        # Visualize the model if possible
        # If you have a predefined model class, you can visualize it:
        # from your_model_module import YourModelClass
        # model_instance = YourModelClass()
        # model_instance.load_state_dict(model)
        # dummy_input = torch.randn(1, 3, 224, 224)  # Adjust input size as necessary
        # st.graphviz_chart(make_dot(model_instance(dummy_input), params=dict(model_instance.named_parameters())).source)
        
        st.write("Visualizing model structure (if supported)...")
        try:
            from your_model_module import YourModelClass  # Replace with actual model import
            model_instance = YourModelClass()  # Replace with actual model initialization
            model_instance.load_state_dict(model)
            dummy_input = torch.randn(1, 3, 224, 224)  # Adjust input size as necessary
            dot = make_dot(model_instance(dummy_input), params=dict(model_instance.named_parameters()))
            st.graphviz_chart(dot.source)
        except Exception as e:
            st.error(f"Error visualizing model: {e}")
    else:
        st.error("Unsupported file format.")

def app():
    st.title("Interactive File and Folder Viewer")

    base_dir = "./results"

    if "current_path" not in st.session_state:
        st.session_state["current_path"] = base_dir

    if "open_file" not in st.session_state:
        st.session_state["open_file"] = None

    # Sidebar settings
    st.sidebar.title("Settings")
    headless = st.sidebar.checkbox("HEADLESS", value=True)
    multigpu = st.sidebar.checkbox("MULTIGPU", value=False)
    enable_recording = st.sidebar.checkbox("ENABLE RECORDING", value=False)

    # Conflict handling for sidebar options
    if multigpu and enable_recording:
        st.sidebar.error("MULTIGPU and ENABLE RECORDING cannot be selected together.")
        enable_recording = False

    def navigate_to(directory):
        st.session_state["current_path"] = directory
        st.session_state["open_file"] = None

    def toggle_file_display(file_path):
        if st.session_state["open_file"] == file_path:
            st.session_state["open_file"] = None
        else:
            st.session_state["open_file"] = file_path

    def display_file_content(file_path):
        _, ext = os.path.splitext(file_path)
        if ext in [".txt", ".py", ".log"]:
            with open(file_path, "r") as file:
                content = file.read()
                st.code(content, language=ext[1:])
        elif ext in [".png", ".jpg", ".jpeg", ".gif"]:
            image = Image.open(file_path)
            st.image(image, caption=file_path)
        elif ext in [".mp4", ".avi", ".mov"]:
            video_file = open(file_path, "rb")
            video_bytes = video_file.read()
            st.video(video_bytes)
        elif ext == ".pth":
            model = torch.load(file_path)
            if isinstance(model, dict) and 'model_state_dict' in model:
                st.write("Model state_dict keys:")
                st.write(list(model['model_state_dict'].keys()))
            else:
                st.write("Model keys:")
                st.write(list(model.keys()))
            
            st.write("Visualizing model structure (if supported)...")
            try:
                from your_model_module import YourModelClass  # Replace with actual model import
                model_instance = YourModelClass()  # Replace with actual model initialization
                model_instance.load_state_dict(model)
                dummy_input = torch.randn(1, 3, 224, 224)  # Adjust input size as necessary
                dot = make_dot(model_instance(dummy_input), params=dict(model_instance.named_parameters()))
                st.graphviz_chart(dot.source)
            except Exception as e:
                st.error(f"Error visualizing model: {e}")
        elif ext in [".yaml", ".yml"]:
            with open(file_path, "r") as file:
                content = file.read()
                st.code(content, language="yaml")
        elif ext in [".json"]:
            with open(file_path, "r") as file:
                content = file.read()
                st.json(content)
        elif ext in [".pkl", ".pickle"]:
            try:
                import pandas as pd
                data = pd.read_pickle(file_path)
                st.write(data)
            except Exception as e:
                st.error(f"Error loading pickle file: {e}")
        else:
            st.error("Unsupported file format.")

    st.sidebar.title("Navigation")
    if st.sidebar.button("Go to Base Directory"):
        navigate_to(base_dir)

    parts = os.path.relpath(st.session_state["current_path"], base_dir).split(os.sep)
    path_accum = base_dir
    for i, part in enumerate(parts):
        if st.sidebar.button(f"{'→' * i} {part}"):
            path_accum = os.path.join(base_dir, *parts[: i + 1])
            navigate_to(path_accum)
            break

    current_files = []
    current_dirs = []
    for item in sorted(os.listdir(st.session_state["current_path"])):
        item_path = os.path.join(st.session_state["current_path"], item)
        if os.path.isdir(item_path):
            current_dirs.append(item)
        else:
            current_files.append(item)

    for directory in sorted(current_dirs):
        if st.button(f"📁 {directory}", key=directory):
            navigate_to(os.path.join(st.session_state["current_path"], directory))

    for file in sorted(current_files):
        if st.button(f"📄 {file}", key=file):
            toggle_file_display(os.path.join(st.session_state["current_path"], file))

        if st.session_state["open_file"] == os.path.join(
            st.session_state["current_path"], file
        ):
            if file.endswith(".py") and "rewardonly" not in file:
                file_content = open(
                    os.path.join(st.session_state["current_path"], file)
                ).read()
                modified_content = st.text_area(
                    f"Modify {file}",
                    value=file_content,
                    height=300,
                    key=f"text_area_{file}",
                )

                if st.button("💾 Save Changes", key=f"save_{file}"):
                    with open(
                        os.path.join(st.session_state["current_path"], file), "w"
                    ) as f:
                        f.write(modified_content)
                    st.success("Changes saved.")

                if st.button("▶️ Run Code", key=f"run_{file}"):
                    with st.spinner(f"Running {file}..."):
                        test_with_this_file(
                            os.path.join(st.session_state["current_path"], file),
                            headless,
                            enable_recording=enable_recording,
                            multigpu=multigpu,
                        )
                        st.success("Run completed.")
            else:
                display_file_content(
                    os.path.join(st.session_state["current_path"], file)
                )

    if st.session_state["open_file"]:
        display_file_content(st.session_state["open_file"])

if __name__ == "__main__":
    st.set_page_config(page_title="File Explorer", layout="wide")
    app()

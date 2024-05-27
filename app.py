import os
import streamlit as st
import subprocess
import yaml
import sys
import shutil
from loguru import logger
from utils.system import check_system_encoding

ENABLE_RECORDING = False
MULTIGPU = False
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


yaml_path = f"./config/{config_name}.yaml"
cfg = load_config(yaml_path)

ISAAC_ROOT_DIR = cfg.gym.omniisaacsimpathenv
PYTHON_PATH = cfg.gym.pythonpath
SCRIPTS_DIR = cfg.gym.scriptpath
TASK_PATH = cfg.output.overwrite  # Assuming TASK_PATH is defined in the YAML config
TASK = cfg.gym.task
BASE_DIR = "./tests"
os.makedirs(BASE_DIR, exist_ok=True)

def test_with_this_file(file_path, headless):
    response_id = file_path.split("/")[-2]
    iter_num = file_path.split("/")[-3].split("iter")[-1]
    date = file_path.split("/")[1]
    os.makedirs(f"{BASE_DIR}/{date}_env_iter{iter_num}_response{response_id}_train", exist_ok=True)
    std_path = f"{BASE_DIR}/{date}_env_iter{iter_num}_response{response_id}_train/omniverse.log"

    shutil.copyfile(file_path, TASK_PATH)  # Copy the file to TASK_PATH
    st.toast(f"{file_path} copied to {TASK_PATH}")
    headless_flag = "headless=True" if headless else "headless=False"

    if platform == "win32":
        driver = cfg.gym.omniisaacsimpathenv.split(":")[0]
        command = f"{driver}: & cd {ISAAC_ROOT_DIR} & {PYTHON_PATH} {SCRIPTS_DIR} task={TASK} {headless_flag} "
    elif platform == "linux":
        command = f"cd {ISAAC_ROOT_DIR} && {PYTHON_PATH} {SCRIPTS_DIR} task={TASK} {headless_flag} "
    else:
        logger.error("Unsupported platform!")
        exit()

    st.info(f"Command: {command}")

    
    with open(std_path, 'w') as log_file:
        buffer = []
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        log_placeholder = st.empty()
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()
            if output == '' and error == '' and process.poll() is not None:
                break
            if output:
                buffer.append(output.strip())
                if len(buffer) > 100:
                    buffer.pop(0)
                log_file.write(output)
                log_file.flush()
                log_placeholder.code("\n".join(buffer), language="bash")
            if error:
                buffer.append(f"ERROR: {error.strip()}")
                if len(buffer) > 100:
                    buffer.pop(0)
                log_file.write(error)
                log_file.flush()
                log_placeholder.code("\n".join(buffer), language="bash")
        process.stdout.close()
        process.stderr.close()
        process.wait()

        if process.returncode != 0:
            log_placeholder.error("There was an error running the Python script.")
            st.error("There was an error running the Python script.")


def app():
    st.title("Interactive File and Folder Viewer")

    base_dir = "./results"

    if "current_path" not in st.session_state:
        st.session_state["current_path"] = base_dir

    if "open_file" not in st.session_state:
        st.session_state["open_file"] = None

    # Sidebar option for HEADLESS mode
    st.sidebar.title("Settings")
    headless = st.sidebar.checkbox("HEADLESS", value=True)

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

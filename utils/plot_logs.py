import os
from tqdm import tqdm
from tensorboard_parser import tensorboard_parser
from loguru import logger

def find_and_plot_events_files(root_path):
    # Traverse the directory tree
    for root, dirs, files in tqdm(os.walk(root_path)):
        for file in files:
            if "events" in file:
                file_path = os.path.join(root, file)
                # Check if the file size is 1KB (1024 bytes)
                if os.path.getsize(file_path) <= 2048:
                    logger.info(f"Skipping events file (1KB): {file}")
                    continue
                logger.info(f"Found events file: {file}")
                # Call the function to plot the log
                tb_parser = tensorboard_parser(file_path, save=True, plot=False, dir_path="plots", name=file)
                tb_parser.parse_and_plot()

# Example usage
root_path = r'H:\Omniverse\Library\isaac_sim-2023.1.1\SimpleEureka\results\2024-05-22_01-41-24'
find_and_plot_events_files(root_path)

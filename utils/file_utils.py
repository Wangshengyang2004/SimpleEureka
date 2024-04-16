import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

def find_files_with_substring(directory, substring):
    matches = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if substring in file:
                matches.append(os.path.join(root, file))
    return matches

def load_tensorboard_logs(path):
    data = defaultdict(list)
    event_acc = EventAccumulator(path)
    event_acc.Reload()  # Load all data written so far

    for tag in event_acc.Tags()["scalars"]:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append(event.value)
    
    return data

import importlib.util

def import_class_from_file(file_path, function_name):
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    function = getattr(module, function_name)
    return function

def inject_back(filepath:str, text:str, marker:str):
    """Inject the Eureka generated functions back to the original python files,
    Will only overwrite functions with the same name and add new functions before def calculate_metrics

    Args:
        filepath (str): _description_
        text (str): _description_
        marker (str): _description_
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    for i, line in enumerate(lines):
        if marker in line:
            lines.insert(i+1, text)
            break
    
    with open(filepath, 'w') as file:
        file.write("".join(lines))
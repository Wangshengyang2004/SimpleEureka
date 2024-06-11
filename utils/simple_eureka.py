import os
import re
import openai
from loguru import logger

def clean_response(text):
    # Remove import statements
    text = re.sub(r'^\s*import .*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*from .* import .*$', '', text, flags=re.MULTILINE)

    # Remove all function definitions except magic methods like __init__, __str__, etc.
    text = re.sub(r'^\s*def (?!(?:__\w+__)\s*\().*$', '', text, flags=re.MULTILINE)

    # Find and remove everything after the last set of triple backticks
    last_backticks = text.rfind('```')
    if last_backticks != -1:
        text = text[:last_backticks]

    # Remove all triple backticks
    text = text.replace('```', '')

    # Clean up extra newlines
    text = re.sub(r'\n\s*\n', '\n', text)  # Reduce multiple newlines to single ones

    return text.strip()


def remove_old_functions(code, function_names):
    for func_name in function_names:
        # Pattern to match the function and everything up until the start of the next function or end of the class
        pattern = r"^\s*def " + re.escape(func_name) + r".*?(?=\n\s*def |\Z)"
        # Replace the pattern with an empty string
        code = re.sub(pattern, "", code, flags=re.MULTILINE | re.DOTALL)
    return code

def final_cleaner(code):
        # Remove the #END sign
        code = code.replace("#END", "")
        code = code.replace("```", "")
        # Remove the continue sign
        code = code.replace("```python", "")
        code = code.replace("python", "")
        return code

def replace_observation_buffer(code):
    # Replace the observation buffer with a placeholder
    code = re.sub(r"observation_buffer = np.zeros\(.*?\)", "observation_buffer = np.zeros(0)", code)
    return code


def extract_and_remove_dict(code_text):
    # Regex pattern to capture and remove dictionary content after 'self.episode_sums ='
    pattern = r"(self\.episode_sums\s*=\s*\{)([^}]+)(\})"
    match = re.search(pattern, code_text, re.DOTALL)
    if match:
        # Extract dictionary content
        dict_content = match.group(2)  # Only the content inside the braces
        # Remove the dictionary from the code
        modified_code = re.sub(pattern, '', code_text, flags=re.DOTALL)
        return dict_content, modified_code
    return None, code_text

def inject_new_dict(partial_code, original_code):
    # Extract the new dictionary content from the partial code and remove it
    new_dict_content, partial_code_without_dict = extract_and_remove_dict(partial_code)
    if new_dict_content is None:
        return original_code, partial_code  # Return original if no new dict found
    
    # Pattern to find and replace the old dictionary in the original code
    old_dict_pattern = r"(self\.episode_sums\s*=\s*\{)[^}]+(\})"
    
    # Replace the old dictionary content with the new one in the original code
    new_original_code = re.sub(old_dict_pattern, r"\1" + new_dict_content + r"\2", original_code, flags=re.DOTALL)
    
    return new_original_code, partial_code_without_dict



def add_imports(original_code: str, generated_code: str):
    # Compare the imported packages of generated code and Add the new imports to the top of the file
    # Extract original imports
    original_imports = re.findall(r'^\s*import .*$', original_code, flags=re.MULTILINE)
    # Remove original imports
    original_code = re.sub(r'^\s*import .*$', '', original_code, flags=re.MULTILINE)
    # New imports
    new_imports = re.findall(r'^\s*import .*$', generated_code, flags=re.MULTILINE)
    # Use a set to avoid duplicates
    imports = set(original_imports + new_imports)
    imports = "\n".join(imports)
    # logger.debug(f"Merged imports:{imports}")
    return imports + "\n" + original_code

def process_response(response, original_crazyflie_code):
    if "END" in response:
        logger.success("Successfully parsed response with #END")
    else:
        logger.error("Error: #END not found in the response or no response received.")
        return None, None

    # Extract code snippets up to the '#END' marker
    resp = response.split("#END")[0]
    original_crazyflie_code, resp = inject_new_dict(resp, original_crazyflie_code)
    # Extract function names using regular expression
    pattern = r"def\s+(\w+)\s*\("
    functions = re.findall(pattern, resp)

    # Remove old functions and add new ones
    updated_code = remove_old_functions(original_crazyflie_code, functions)
    updated_code = add_imports(updated_code, resp)
    new_code_snippets = "\n".join(resp.split("\n"))
    updated_code += "\n    # New Code Starts Here\n" + new_code_snippets
    updated_code = final_cleaner(updated_code)

    return updated_code, new_code_snippets

def task_description_optimizer(cfg, task_description):
    messages = [
        {
            "role": "system",
            "content": task_description
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
    response = response_cur.choices[0].message.content
    return response

if __name__ == "__main__":
    # Example usage:
    partial_code = """
    # Import necessary libraries
    import torch

    self.episode_sums = {
        "rew_pos": torch.zeros(),
        "rew_orient": torch.zeros(),
        "new_metric": torch.zeros(),  # This is a new entry
    }

    # Some additional unrelated code
    def additional_function():
        pass
    """

    original_code = """
    class EpisodeMetrics:
        def __init__(self):
            self.episode_sums = {
                "rew_pos": torch.zeros(),
                "rew_orient": torch.zeros(),
                "rew_effort": torch.zeros(),
            }
    """

    updated_code, modified_partial_code = inject_new_dict(partial_code, original_code)
    print("Updated Original Code:\n", updated_code)
    print("Modified Partial Code:\n", modified_partial_code)
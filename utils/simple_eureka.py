import re

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
    return imports + "\n" + original_code
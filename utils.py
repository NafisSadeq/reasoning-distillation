import re

def extract_list_substring(input_string):

    pattern = r'\[.*?\]'
    match = re.search(pattern, input_string)
    if match:
        return match.group(0)
    return None
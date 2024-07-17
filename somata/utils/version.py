import re


def extract_version(file_path):
    # Open the readme file
    with open(file_path, 'r') as file:
        content = file.read()

    # Define the regex pattern to find the version number
    version_pattern = re.compile(r'Version-([0-9]+\.[0-9]+\.[0-9]+(?:[a-zA-Z0-9]*)?)-(orange|green)')

    # Search for the pattern in the content
    match = version_pattern.search(content)

    # If a match is found, return the version number
    if match:
        return match.group(1)
    else:
        return None


# Extract the version number
VERSION = extract_version("README.md")

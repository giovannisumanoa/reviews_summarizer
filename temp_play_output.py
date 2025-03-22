import json
import re

def extract_and_validate_json(raw_response):
    """
    Extracts and validates JSON content from a raw LLM response.

    Args:
        raw_response (str): The raw text output from the LLM.

    Returns:
        dict or None: Parsed JSON dictionary if valid, else None.
    """
    if not raw_response or not isinstance(raw_response, str):
        print("Warning: Empty or invalid response received.")
        return None

    # Extract JSON-like content using regex
    match = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if match:
        raw_response = match.group(0)  # Extract only JSON content

    # Fix potential formatting issues
    raw_response = raw_response.replace("”", "\"").replace("“", "\"").strip()  # Normalize quotes

    # Attempt to parse JSON
    try:
        parsed_json = json.loads(raw_response)
    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e}")
        print("Raw response that failed:\n", raw_response)
        return None

    # Ensure the parsed data is a dictionary
    if not isinstance(parsed_json, dict):
        print("Warning: Parsed JSON is not a dictionary.")
        return None

    return parsed_json

# Step 1: Load the content of the text file
with open("raw_llm_response.txt", "r", encoding="utf-8") as file:
    raw_response = file.read().strip()  # Read and remove extra whitespace

# Extract and parse JSON response
try:
    structured_response = extract_and_validate_json(raw_response)
except json.JSONDecodeError:
    print("Warning: Could not parse response as JSON. Returning empty structure.")
    structured_response = {"pros": {}, "cons": {}, "suggestions": {}}

# Print the extracted structured response
print(structured_response)
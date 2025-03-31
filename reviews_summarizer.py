
import json
import os
import re
from collections import Counter
from collections import defaultdict
from typing import Optional, Dict, Any
from datetime import datetime
from openai import OpenAI, OpenAIError
from tqdm import tqdm  # For progress bar

LLM_BASE_URL = "https://api.deepseek.com"
LLM_API_KEY = "sk-b49e0cc821cc41cea896e5fa21322c90"
LLM_MODEL = "deepseek-chat"
#LLM_MODEL = "deepseek-reasoner"


client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def log_info(info: str, log_dir: str):
    """
    Log information to a file with timestamp.

    Args:
        info (str): The information to log
        log_dir (str): Directory where log file will be stored
    """
    log_file = os.path.join(log_dir, "log.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}]\n{info}\n\n"

    # Create directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

def generate_folder_name():
    """
    Generate a folder name based on the current date and time (up to minutes).

    Returns:
        str: A formatted string for the folder name.
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M")

def get_llm_response(
    user_prompt: str,
    system_prompt: str = None,
    model_params: Dict[str, Any] = None,
    log_dir: str = None
):
    """
    Get a response from the LLM for the given prompts.

    Args:
        user_prompt (str): The user's input message.
        system_prompt (str): The system message providing context.
        log_dir (str, optional): Directory to log request parameters.
        model_params (dict, optional): Dictionary of model parameters, e.g.:
            {
                "temperature": 0.7,
                "max_tokens": 100,
                "response_format": "json",
                "top_p": 0.9,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.1
            }

    Returns:
        str: The LLM response object.

    Raises:
        ValueError: If the LLM request fails.
    """
    messages = [{"role": "user", "content": user_prompt}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    params = {
        "model": LLM_MODEL,
        "messages": messages
    }

    # Merge optional model parameters if provided
    if model_params:
        params.update(model_params)
        # Ensure response_format is wrapped properly if provided
        if "response_format" in model_params:
            params["response_format"] = {"type": model_params["response_format"]}

    print_as_json(params)

    # Log the request parameters if logging is enabled
    if log_dir:
        info = json.dumps(params, indent=2)
        log_info(info, log_dir)

    try:
        response = client.chat.completions.create(**params)
    except OpenAIError as e:
        raise ValueError(f"LLM request failed: {e}") from e

    return response.choices[0].message.content

def print_as_json(data, indent=4, ensure_ascii=False, sort_keys=False):
    """
    Prints a dictionary or JSON-compatible data in a human-readable JSON format.

    Args:
        data: The dictionary/JSON data to print
        indent: Number of spaces for indentation (default: 4)
        ensure_ascii: If False, preserves non-ASCII characters (default: False)
        sort_keys: If True, sorts dictionary keys alphabetically (default: False)
    """
    print(json.dumps(
        data,
        indent=indent,
        ensure_ascii=ensure_ascii,
        sort_keys=sort_keys
    ))

def load_reviews(file_path, batch_size=500):
    """Loads reviews from JSON and chunks them into batches of 500."""
    with open(file_path, 'r', encoding='utf-8') as f:
        reviews = json.load(f)

    # Extract only review text
    review_texts = [r["text"] for r in reviews]

    # Chunk into batches of 500 reviews each
    batches = [review_texts[i:i + batch_size] for i in range(0, len(review_texts), batch_size)]

    return batches

def print_batches(batches, num_batches=0, items_per_batch=0):
    """
    Prints a specified number of batches and reviews per batch from the given batches.
    If num_batches or items_per_batch exceeds available data, prints as much as possible.
    If either parameter is set to 0, it prints all available data for that parameter.

    Parameters:
    - batches: List of lists containing reviews (batches of reviews).
    - num_batches: The number of batches to print. Set to 0 to print all batches.
    - items_per_batch: The maximum number of items to print per batch. Set to 0 to print all items in each batch.
    """
    # If num_batches is 0, print all available batches
    if num_batches == 0:
        num_batches = len(batches)

    # Loop over the specified number of batches
    for idx in range(num_batches):
        batch = batches[idx]

        # If reviews_per_batch is 0, print all reviews in the current batch
        if items_per_batch == 0:
            items_to_print = batch
        else:
            # Otherwise, print up to the specified number of reviews
            items_to_print = batch[:items_per_batch]

        print(f"--- Batch {idx + 1} ---")

        for i, item in enumerate(items_to_print):
            print(f"Item {i + 1}: {item}")

        print("\n")  # Add a newline for separation between batches

def extract_json(raw_response):
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

def generate_next_filename(folder, base_name):
    """Finds the next available filename in the sequence raw_llm_response_{idx}.txt inside the specified folder."""
    os.makedirs(folder, exist_ok=True)  # Ensure the directory exists
    existing_files = [f for f in os.listdir(folder) if f.startswith(f"{base_name}_") and f.endswith(".txt")]

    indices = []
    for filename in existing_files:
        try:
            index = int(filename.split("_")[-1].split(".")[0])
            indices.append(index)
        except ValueError:
            continue  # Ignore files that don't fit the expected naming pattern

    next_index = max(indices, default=0) + 1
    return os.path.join(folder, f"{base_name}_{next_index}.txt")

def analyze_reviews(review_batch, output_dir, base_name="raw_llm_response"):
    """Asks LLM to analyze likes, dislikes, and suggestions in the reviews and counting similar points and saves results."""
    system_prompt = "You are a helpful assistant"
    user_prompt = f"""
    Analyze these customer reviews of a payments gateway company and extract:
    - What customers like (Pros)
    - What customers dislike (Cons)
    - Suggestions for improvement or new features

    Reviews:
    {chr(10).join(review_batch)}

    Format your response in strict JSON format, ensuring that similar points are grouped and counted:
    {{
      "pros": {{"Point 1": count, "Point 2": count}},
      "cons": {{"Point 1": count, "Point 2": count}},
      "suggestions": {{"Point 1": count, "Point 2": count}}
    }}
    """

    raw_response = get_llm_response(system_prompt, user_prompt)

    # Save raw response to a new uniquely named file in the specified folder
    filename = generate_next_filename(output_dir, base_name)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(raw_response)


def process_batches(batches, output_dir, num_batches=0):
    """Processes a specified number of batches. If num_batches is 0 or exceeds available batches, process all."""

    # Determine the actual number of batches to process
    if num_batches == 0 or num_batches > len(batches):
        num_batches = len(batches)

    # Process only the selected number of batches
    for batch in tqdm(batches[:num_batches], desc="Processing Batches"):
        analyze_reviews(batch, output_dir)

def aggregate_similar_points(folder):
    """Reads raw response files, extracts structured data, and consolidates summaries."""
    all_pros = Counter()
    all_cons = Counter()
    all_suggestions = Counter()

    existing_files = [f for f in os.listdir(folder) if f.startswith("raw_llm_response_") and f.endswith(".txt")]

    for filename in existing_files:
        file_path = os.path.join(folder, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            raw_response = file.read()

        # Extract and parse JSON response
        try:
            structured_response = extract_json(raw_response)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {filename} as JSON. Skipping.")
            continue

        # Update counters with occurrences
        all_pros.update(structured_response.get("pros", {}))
        all_cons.update(structured_response.get("cons", {}))
        all_suggestions.update(structured_response.get("suggestions", {}))

    return {
        'pros': dict(all_pros),
        'cons': dict(all_cons),
        'suggestions': dict(all_suggestions)
    }

def group_reviews(folder):
    """
    Reads raw response files and returns consolidated data WITHOUT merging similar keys.
    Returns a dictionary where each key contains a list of all individual responses.
    """
    consolidated = {
        'pros': defaultdict(list),
        'cons': defaultdict(list),
        'suggestions': defaultdict(list)
    }

    existing_files = [f for f in os.listdir(folder) if f.startswith("raw_llm_response_") and f.endswith(".txt")]

    for filename in existing_files:
        file_path = os.path.join(folder, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            raw_response = file.read()

        try:
            structured_response = extract_json(raw_response)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse {filename} as JSON. Skipping.")
            continue

        # Append all values without aggregation
        for category in ['pros', 'cons', 'suggestions']:
            for key, value in structured_response.get(category, {}).items():
                consolidated[category][key].append(value)

    # Convert defaultdict to regular dict for cleaner output
    return {
        'pros': dict(consolidated['pros']),
        'cons': dict(consolidated['cons']),
        'suggestions': dict(consolidated['suggestions'])
    }


def save_dict_to_txt(data_dict, output_dir, file_name):
    """
    Save a dictionary to a text file in JSON format.

    Args:
        data_dict (dict): Dictionary to be saved
        output_dir (str): Directory where the file will be saved
        file_name (str): Name of the output file (without extension)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create full file path
    file_path = os.path.join(output_dir, f"{file_name}.txt")

    # Save dictionary to file with pretty formatting
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=4)

    print(f"Dictionary successfully saved to {file_path}")


def consolidate_similar_entries(data, output_dir):
    """
    Consolidates similar entries in a dictionary by grouping semantically similar keys and summing their values.

    Args:
        data (dict): Dictionary with 'pros', 'cons', and 'suggestions' keys
    """

    # Define system and user prompts as separate variables
    system_prompt = """You are a customer feedback analyst specializing in semantic grouping. 
    Your task is to identify phrases that convey identical or nearly identical meanings, 
    then combine them by selecting the clearest phrasing and summing their counts.
    Always return perfect JSON without additional commentary."""

    consolidated = {"pros": {}, "cons": {}, "suggestions": {}}

    first_call = True

    for category in tqdm(data.keys(), desc="Processing categories"):
        if not data[category]:  # Skip empty categories
            continue

        # Prepare the user prompt
        user_prompt = f"""Please analyze the following list of {category} passed as a JSON dictionary. 
        Each key represents a comment from customers regarding the services from their payments gateway provider.
        The key's corresponding value represents the count of customers that made such (or a similar) comment. 
        Please group together comments that express the same or very similar sentiments, by creating a consolidated 
        comment using a representative phrase that best captures the grouped comments meaning.
         
        It is very important that you sum the counts of the grouped comments and assign the result as the
        count of the consolidated comment. e.g. if the data includes the comments
        
        "Hidden fees and unexpected charges": 10,
        "I was charged unexpected charges and fees": 10,
        
        then you should consolidate this comment and output something like 
        
        "Hidden fees and unexpected charges": 20
        
        Return ONLY a JSON dictionary where keys are the consolidated comments and values the corresponding summed counts.
        Do not include any additional commentary or explanation.

        Here is the input data:
        {json.dumps(data[category], indent=2)}
        """
        single_prompt = final_prompt = f"{system_prompt}\n\n{user_prompt}\n"
        print(single_prompt)
        if first_call:
            log_dir = output_dir
        else:
            log_dir = None
            first_call = False

        try:
            # Call the LLM
            raw_response = get_llm_response(
                user_prompt,
                system_prompt,
                {
                #     "temperature": 0.7,
                #     "max_tokens": 1000,
                     "response_format": "json_object"
                #     "top_p": 1,
                #     "frequency_penalty": 0.0,
                #     "presence_penalty": 0.0
                },
                log_dir
            )

            filename = generate_next_filename(output_dir, "llm_response_def_params")
            # Save raw responses immediately
            with open(filename, "w", encoding="utf-8") as file:
                file.write(raw_response)

            # Parse the response
            result = extract_json(raw_response)
            consolidated[category] = result

            first_call = False

        except Exception as e:
            print(f"Error processing {category}: {str(e)}")
            #consolidated[category] = data[category]  # Fallback to original if error occurs

    print_as_json(consolidated)
    
    save_dict_to_txt(consolidated, output_dir, "consolidated_reviews_def_params")


# Define your main function
def main():
    run_dir = generate_folder_name()
    batch_dir = os.path.join("llm_responses", run_dir)
    consolidated_dir = os.path.join("consolidated_responses", run_dir)
    # Create reviews batches
    batches = load_reviews("trustpilot_reviews.json", 80)
    # Print total number of batches
    print(f"Number of batches: {len(batches)}\n")
    # Print a limited number of batches and items per batch for visual exploration
    #print_batches(batches, 5, 5)
    # Process the batches and save the responses
    #process_batches(batches, batch_dir, 4)
    # Load the responses and add up similar points
    aggregates = aggregate_similar_points("llm_responses")
    # Print the points and their count
    print_as_json(aggregates)
    # save aggregated reviews
    save_dict_to_txt(aggregates, batch_dir, "aggregated_reviews")
    #result_grouped = group_reviews(batch_dir)
    #print_as_json(result_grouped)
    consolidate_similar_entries(aggregates, consolidated_dir)

# Call main() at the bottom
if __name__ == "__main__":
    main()
import os
import re
import json
from openai import OpenAI
from openai import batches
from collections import Counter
from collections import defaultdict
from tqdm import tqdm  # For progress bar

LLM_BASE_URL = "https://api.deepseek.com"
LLM_API_KEY = "sk-b49e0cc821cc41cea896e5fa21322c90"
LLM_MODEL = "deepseek-chat"


client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def get_llm_response(client, prompt):
    """
    Get a response from the LLM for the given prompt.

    Args:
        client: The initialized LLM client
        prompt (str): The input prompt to send to the LLM

    Returns:
        The LLM response object
    """
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        stream=False
    )
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

def get_next_filename(folder):
    """Finds the next available filename in the sequence raw_llm_response_{idx}.txt inside the specified folder."""
    os.makedirs(folder, exist_ok=True)  # Ensure the directory exists
    existing_files = [f for f in os.listdir(folder) if f.startswith("raw_llm_response_") and f.endswith(".txt")]

    indices = []
    for filename in existing_files:
        try:
            index = int(filename.split("_")[-1].split(".")[0])
            indices.append(index)
        except ValueError:
            continue  # Ignore files that don't fit the expected naming pattern

    next_index = max(indices, default=0) + 1
    return os.path.join(folder, f"raw_llm_response_{next_index}.txt")

def analyze_reviews(review_batch, output_dir):
    """Asks LLM to analyze likes, dislikes, and suggestions in the reviews and counting similar points and saves results."""
    prompt = f"""
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

    raw_response = get_llm_response(client, prompt)

    # Save raw response to a new uniquely named file in the specified folder
    filename = get_next_filename(output_dir)
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

def add_up_similar_points(folder):
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
            structured_response = extract_and_validate_json(raw_response)
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

def count_similar_reviews(folder):
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
            structured_response = extract_and_validate_json(raw_response)
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

def meta_summarization(summaries):
    """Asks GPT to consolidate batch summaries into a final structured report."""
    prompt = f"""
    Here are summarized customer reviews for a payments gateway company, grouped into batches.
    Your task is to analyze and consolidate similar points to create a final structured summary.

    Please:
    - Merge similar points under a single, well-phrased insight.
    - Eliminate redundant or overly similar points.
    - Structure the response into three categories: Pros, Cons, and Suggestions.

    Batch Summaries:
    {chr(10).join(summaries)}

    Format response as:
    **Pros:**
    - Final merged insight 1
    - Final merged insight 2

    **Cons:**
    - Final merged insight 1
    - Final merged insight 2

    **Suggestions:**
    - Final merged insight 1
    - Final merged insight 2
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        stream=False
    )

    raw_response = response.choices[0].message.content

# Define your main function
def main():
    #batch_dir = "processed_batches"
    batch_dir = "llm_responses"
    # Create reviews batches
    batches = load_reviews("trustpilot_reviews.json", 80)
    # Print total number of batches
    print(f"Number of batches: {len(batches)}")
    # Print a limited number of batches and items per batch for visual exploration
    print_batches(batches, 5, 5)
    print()
    # Process the batches and save the responses
    #process_batches(batches, batch_dir, 4)
    # Load the responses and add up similar points
    result_add_up = add_up_similar_points(batch_dir)
    # Print the points and their count
    print_as_json(result_add_up)
    result_count = count_similar_reviews(batch_dir)
    print_as_json(result_count)

# Call main() at the bottom
if __name__ == "__main__":
    main()
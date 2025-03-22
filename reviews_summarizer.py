import re
import json
from openai import OpenAI
from openai import batches
from tqdm import tqdm  # For progress bar

BASE_URL = "https://api.deepseek.com"
API_KEY = "sk-b49e0cc821cc41cea896e5fa21322c90"


client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


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


def print_batch_info(list_of_lists):
    """Prints the number of sublists (batches) and the number of items per sublist, assuming all sublists have the same length."""

    # Get the number of sublists (batches)
    num_batches = len(list_of_lists)

    # Get the number of items in the first sublist (assuming all sublists have the same number of items)
    items_per_batch = len(list_of_lists[0]) if num_batches > 0 else 0

    # Print the results
    print(f"Number of batches: {num_batches}")
    print(f"Number of items per batch: {items_per_batch}")


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

def analyze_reviews(review_batch):
    """Asks LLM to analyze likes, dislikes, and suggestions in the reviews and returns structured data with counts."""
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

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        stream=False
    )

    raw_response = response.choices[0].message.content

    # Save raw response to a text file immediately for debugging
    with open("raw_llm_response.txt", "w", encoding="utf-8") as file:
        file.write(raw_response)

    # Extract and parse JSON response
    try:
        structured_response = extract_and_validate_json(raw_response)
    except json.JSONDecodeError:
        print("Warning: Could not parse response as JSON. Returning empty structure.")
        structured_response = {"pros": {}, "cons": {}, "suggestions": {}}

    return structured_response

from collections import Counter

def consolidate_reviews(reviews):
    # Initialize Counter objects to store consolidated data
    all_pros = Counter()
    all_cons = Counter()
    all_suggestions = Counter()

    # Iterate through each review
    for review in reviews:
        # Update the counters for pros, cons, and suggestions
        all_pros.update(review['pros'])
        all_cons.update(review['cons'])
        all_suggestions.update(review['suggestions'])

    # Return the consolidated data as a single JSON structure
    return {
        'pros': dict(all_pros),
        'cons': dict(all_cons),
        'suggestions': dict(all_suggestions)
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
    # Create reviews batches
    batches = load_reviews("trustpilot_reviews.json", 80)
    # Print total number of batches and number of reviews per batch
    print_batch_info(batches)
    # Print a limited number of batches and items per batch for visual exploration
    print_batches(batches, 5, 5)
    print()
    # print(analyze_reviews(batches[0]))



# Call main() at the bottom
if __name__ == "__main__":
    main()
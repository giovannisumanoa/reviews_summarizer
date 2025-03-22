import requests
import time
import random
import json
from bs4 import BeautifulSoup

# Base URL of Trustpilot reviews
BASE_URL = "https://ie.trustpilot.com/review/globalpayments.com?page="
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Set the number of pages to scrape (set to None for all pages)
MAX_PAGES = None  # Change this to your desired limit (or None for unlimited)


def get_page_data(url):
    """Fetches HTML content of the given URL."""
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print("Failed to fetch page:", url)
        return None
    return response.text


def extract_reviews(html):
    """Extracts reviews from the JSON data inside the HTML."""
    soup = BeautifulSoup(html, "html.parser")
    script_tag = soup.find("script", {"type": "application/json"})

    if not script_tag:
        print("JSON data not found in the HTML.")
        return []

    try:
        # json.loads() parses a JSON string into a Python dictionary.
        # json.dumps() converts a dictionary back to a JSON string.
        json_data = json.loads(script_tag.string)
        reviews = json_data["props"]["pageProps"]["reviews"]
        extracted_reviews = []

        for review in reviews:
            extracted_reviews.append({
                "review_id": review["id"],
                "rating": review["rating"],
                "text": review["text"],
                "published_date": review["dates"]["publishedDate"],
                "consumer": {
                    "displayName": review["consumer"]["displayName"],
                    # json.loads() parses a JSON string into a Python dictionary.
                    # .get() prevents this error and allows a default value
                    "countryCode": review["consumer"].get("countryCode", "Unknown"),
                    "isVerified": review["consumer"].get("isVerified", False)
                }
            })

        return extracted_reviews
    except (KeyError, json.JSONDecodeError) as e:
        print("Error extracting reviews:", e)
        return []


def scrape_pages():
    """Loops through paginated pages, extracts reviews, and saves them to JSON."""
    page = 1
    all_reviews = []

    while True:
        if MAX_PAGES and page > MAX_PAGES:  # Stop if max page limit is reached
            print(f"Reached max page limit ({MAX_PAGES}). Stopping.")
            break

        url = BASE_URL + str(page)
        print(f"Scraping page {page}...")

        html = get_page_data(url)
        if not html:
            break  # Stop if request fails

        reviews = extract_reviews(html)
        if not reviews:
            print("No more reviews found. Stopping.")
            break  # Stop if no reviews are found (indicates last page)

        all_reviews.extend(reviews)

        # Check if there's a "Next" button
        soup = BeautifulSoup(html, "html.parser")
        next_page = soup.select_one("nav a[aria-label='Next page']")

        if not next_page:  # If no next button, we are on the last page
            print("Last page reached.")
            break

        # Introduce a random delay before moving to the next page
        delay = random.uniform(1, 2)
        print(f"Waiting for {delay:.2f} seconds before next request...")
        time.sleep(delay)

        page += 1  # Move to the next page

    # Save extracted reviews to a JSON file (overwrites previous data)
    with open("trustpilot_reviews.json", "w", encoding="utf-8") as f:
        json.dump(all_reviews, f, indent=4, ensure_ascii=False)

    print(f"Scraping complete! {len(all_reviews)} reviews saved to trustpilot_reviews.json")


# Run the scraper
scrape_pages()

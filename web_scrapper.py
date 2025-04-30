import json
import os
from scrape_all import WebsiteCrawler
from agentic_extractor import extract

def load_restaurant_data(filename):
    """Load existing restaurant data or return empty list."""
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    return []

def save_restaurant_data(data, filename):
    """Save restaurant data to JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def scrape_and_extract(website_url, temp_file="temp_content.json"):
    """Scrape a website and extract structured data."""
    crawler = WebsiteCrawler(
        base_url=website_url,
        output_file=temp_file.replace(".json", ""),  
        delay=1.0,
        use_selenium=False,
        max_pages=None
    )
    
    try:
        print(f"\n Scraping: {website_url}")
        crawler.crawl()
        
        # Extract structured data
        restaurant_data = extract(temp_file)
        if not restaurant_data:
            print(f"No valid data extracted from {website_url}")
            return None
        
        return restaurant_data
    
    except Exception as e:
        print(f"Failed to process {website_url}: {str(e)}")
        return None
    
    finally:
        crawler.cleanup()
        if os.path.exists(temp_file):
            os.remove(temp_file)  # Delete temp file

if __name__ == "__main__":
    # List of restaurant websites to scrape
    restaurant_websites = [
        "https://www.yangskitchenla.com/",
        "https://www.anjappar.com/",
        "https://www.biryaniblues.com/",
        "https://www.sagarratna.in/",
        "http://www.desivibes.in/"
    ]
    final_output = "restaurant_data.json"
    all_restaurants = load_restaurant_data(final_output)
    
    for url in restaurant_websites:
        data = scrape_and_extract(url)
        if data:  # Only append if extraction succeeded
            all_restaurants.append(data)
            save_restaurant_data(all_restaurants, final_output)
            print(f" Added data from: {url}")
    
    print(f"\n Done! Final data saved to {final_output}")

import os
import json
from apify_client import ApifyClient
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import Optional

# ----------------- CONFIGURATION -----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is missing.")
if not APIFY_API_TOKEN:
    raise ValueError("APIFY_API_TOKEN environment variable is missing.")

ai_client = genai.Client(api_key=GEMINI_API_KEY)
apify_client = ApifyClient(APIFY_API_TOKEN)


# ----------------- DATA SCHEMA -----------------
class LeadEvaluation(BaseModel):
    is_qualified_lead: bool = Field(
        description="True ONLY if the user is an individual/tenant looking for a long-term rental (1 month or longer). False if this is an owner, agent, listing, sublet offer, short holiday stay, or seller."
    )
    summary: str = Field(description="1-line summary of what the person wants.")
    duration: Optional[str] = Field(default=None, description="Rental period (e.g. 1 month, 6 months, 1 year).")
    budget: Optional[str] = Field(default=None, description="Budget in THB or other currency if specified.")
    move_in_date: Optional[str] = Field(default=None, description="Move-in date or timeframe.")
    contact_phone: Optional[str] = Field(default=None, description="Phone number or WhatsApp number if stated.")


# ----------------- 1. SCRAPE FACEBOOK -----------------
def scrape_facebook(group_urls, limit_per_group=5):
    """
    Scrapes target Facebook groups using Apify.
    Default actor: apify/facebook-groups-scraper (public feeds, chronological).
    """
    run_input = {
        "startUrls": [{"url": url} for url in group_urls],
        "resultsLimit": limit_per_group,
        "viewOption": "CHRONOLOGICAL"
    }

    leads = []
    print(f"Triggering Apify Facebook scraper for {len(group_urls)} group(s)...")

    try:
        run = apify_client.actor("apify/facebook-groups-scraper").call(run_input=run_input)
        dataset_id = run.get("defaultDatasetId")
        print(f"Scraper finished with status: {run.get('status')}. Dataset ID: {dataset_id}")

        for item in apify_client.dataset(dataset_id).iterate_items():
            text = item.get("text") or item.get("message", "")
            url = item.get("url") or item.get("postUrl", "https://facebook.com")
            author = item.get("user", {}).get("name") or item.get("authorName", "Unknown")

            if text and len(text.strip()) > 15:
                leads.append({
                    "author": author,
                    "url": url,
                    "text": text.strip()
                })

    except Exception as e:
        print(f"Error scraping Facebook via Apify: {e}")

    return leads


# ----------------- 2. AI EVALUATOR -----------------
def evaluate_lead(post_text: str) -> Optional[LeadEvaluation]:
    prompt = f"""
    Analyze the following Facebook post from a Koh Phangan group.
    Determine if this is a genuine TENANT looking for a LONG-TERM house/villa rental (1 month or longer).

    CRITICAL DISCARD RULES:
    - Return is_qualified_lead = FALSE if this is an owner or agent advertising an available villa, house, room, or sublet.
    - Return is_qualified_lead = FALSE if this is a short holiday or party trip (< 1 month).
    - Return is_qualified_lead = TRUE ONLY if an individual/family is asking to RENT for 1+ months.

    Post Content:
    \"\"\"{post_text}\"\"\"
    """

    try:
        response = ai_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LeadEvaluation,
                temperature=0.1
            ),
        )
        return LeadEvaluation.model_validate_json(response.text)
    except Exception as e:
        print(f"Gemini evaluation error: {e}")
        return None


# ----------------- 3. PIPELINE MAIN -----------------
def main():
    target_groups = [
        "https://www.facebook.com/groups/kohphanganrentals/",
        "https://www.facebook.com/groups/kohphanganpeople/"
    ]

    posts = scrape_facebook(target_groups, limit_per_group=5)
    print(f"\nFetched {len(posts)} posts from Facebook. Evaluating with Gemini Flash...\n")

    qualified_count = 0

    for idx, post in enumerate(posts, 1):
        print(f"[{idx}/{len(posts)}] Checking post by '{post['author']}'...")
        eval_result = evaluate_lead(post["text"])

        if eval

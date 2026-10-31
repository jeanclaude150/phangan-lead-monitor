#!/usr/bin/env python3
"""
Koh Phangan Rental Lead Scanner (monitor.py)
Scrapes target Facebook groups via Apify, filters rental listings using Gemini,
and dispatches an alert email digest via Resend.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from apify_client import ApifyClient
from google import genai
from google.genai import types
import resend

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("phangan-monitor")

# --- Configuration & Defaults ---
APIFY_ACTOR_ID = "apify/facebook-groups-scraper"
DEFAULT_CACHE_FILE = Path("scraped_posts_cache.json")
DEFAULT_MAX_POSTS_PER_GROUP = 20

# Koh Phangan Group targets (add or modify group URLs here)
TARGET_GROUPS = [
    "https://www.facebook.com/groups/1220487088900101/",
    "https://www.facebook.com/share/g/1DWFxdTZGt/",
    "https://www.facebook.com/share/g/1FDBixkSVE/",
    "https://www.facebook.com/share/g/1E1rtUvLfV/",
    "https://www.facebook.com/share/g/1MNnZCTotc/",
    "https://www.facebook.com/share/g/1DYeuoRafj/",
    "https://www.facebook.com/share/g/1CrKqPsbSs/",
    "https://www.facebook.com/share/g/1EgVmuFSbz/",
    "https://www.facebook.com/share/g/14mjdRXMAVG/",
    "https://www.facebook.com/share/g/1KgCDeNmwr/",
    "https://www.facebook.com/share/g/1E4B29pfNQ/",
    "https://www.facebook.com/share/g/1FShqFuiK6/",
    "https://www.facebook.com/share/g/1H3kfwcedc/",
    "https://www.facebook.com/share/g/1bpzE1TPGs/",
    "https://www.facebook.com/share/g/1Jr7JUrsw7/",
    "https://www.facebook.com/share/g/1BXmRFS45J/",
    "https://www.facebook.com/share/g/1J6QnQBWMu/",
    "https://www.facebook.com/share/g/1GRSz7thgg/",
    "https://www.facebook.com/share/g/1EtCDMcLJA/",
    "https://www.facebook.com/share/g/19jMLi1nXR/",
    "https://www.facebook.com/share/g/19ttApseXv/",
    "https://www.facebook.com/share/g/1CQwXhSYoC/",
    "https://www.facebook.com/share/g/1Mr7GVvsTD/",
]


def resolve_dataset_id(run_obj: Any) -> str:
    """
    Safely retrieves the defaultDatasetId from Apify Run objects or dicts.
    Handles 'Run' model instances as well as legacy dict responses.
    """
    if hasattr(run_obj, "default_dataset_id") and run_obj.default_dataset_id:
        return run_obj.default_dataset_id
    if hasattr(run_obj, "defaultDatasetId") and run_obj.defaultDatasetId:
        return run_obj.defaultDatasetId
    if isinstance(run_obj, dict):
        return run_obj.get("defaultDatasetId") or run_obj.get("default_dataset_id")
    raise ValueError(f"Could not extract default dataset ID from Run representation: {type(run_obj)}")


def fetch_posts(
    apify_client: ApifyClient,
    run_id: Optional[str] = None,
    use_cache: bool = False,
    cache_path: Path = DEFAULT_CACHE_FILE,
    max_posts: int = DEFAULT_MAX_POSTS_PER_GROUP,
) -> List[Dict[str, Any]]:
    """
    Retrieves scraped posts from local cache, an existing Apify run, or a fresh scrape.
    """
    # 1. Check local file cache
    if use_cache and cache_path.exists():
        logger.info(f"Loading posts from local cache: {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 2. Existing run ID bypass (no new scraper cost)
    if run_id:
        logger.info(f"Connecting to existing Apify run: {run_id}")
        run = apify_client.run(run_id).get()
        dataset_id = resolve_dataset_id(run)
    else:
        # 3. New fresh scraper run
        logger.info(f"Starting fresh Apify scrape for {len(TARGET_GROUPS)} groups (limit: {max_posts})...")
        run_input = {
            "startUrls": [{"url": url} for url in TARGET_GROUPS],
            "resultsLimit": max_posts,
            "maxPosts": max_posts,
        }
        run = apify_client.actor(APIFY_ACTOR_ID).call(run_input=run_input)
        dataset_id = resolve_dataset_id(run)

    logger.info(f"Retrieving items from Apify dataset: {dataset_id}")
    dataset_client = apify_client.dataset(dataset_id)
    items = list(dataset_client.iterate_items())

    # Update local cache for subsequent rapid iterations
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        logger.info(f"Cached {len(items)} items to {cache_path}")
    except IOError as e:
        logger.warning(f"Failed to cache items locally: {e}")

    return items


def filter_leads_with_gemini(posts: List[Dict[str, Any]], gemini_api_key: str) -> List[Dict[str, Any]]:
    """
    Uses Gemini 2.5 Flash to batch evaluate posts and isolate rental opportunities.
    """
    client = genai.Client(api_key=gemini_api_key)

    # Clean and consolidate posts into lightweight text payloads
    cleaned_posts = []
    for idx, post in enumerate(posts):
        text = post.get("text") or post.get("postText") or post.get("message") or ""
        url = post.get("url") or post.get("postUrl") or post.get("canonicalUrl") or ""
        time_str = post.get("time") or post.get("timestamp") or ""
        
        if len(text.strip()) < 20:  # Skip empty or negligible posts
            continue

        cleaned_posts.append({
            "id": idx,
            "text": text[:1500],  # Truncate overly long text
            "url": url,
            "timestamp": time_str
        })

    if not cleaned_posts:
        logger.info("No substantial posts found to evaluate.")
        return []

    logger.info(f"Evaluating {len(cleaned_posts)} candidate posts with Gemini...")

    prompt = (
        "You are an assistant filtering Koh Phangan real estate leads.\n"
        "Analyze the following list of Facebook group posts and identify properties available for rent.\n"
        "Filter for house, villa, bungalow, or apartment rentals (especially long-term or monthly).\n"
        "Exclude: 'ISO/Looking for' posts (people seeking rentals), general spam, services, scooter rentals, and sales.\n\n"
        "Return a pure JSON list of objects matching this schema:\n"
        "[\n"
        "  {\n"
        "    \"post_id\": <integer id matching input>,\n"
        "    \"is_rental_listing\": true,\n"
        "    \"summary\": \"concise description of the property, bedrooms, location\",\n"
        "    \"price_thb\": \"monthly price if specified or 'Unknown'\",\n"
        "    \"location\": \"specific Koh Phangan area (e.g. Sri Thanu, Hin Kong, Thong Sala, Haad Salad) if mentioned or 'Unspecified'\"\n"
        "  }\n"
        "]\n\n"
        f"Input Posts:\n{json.dumps(cleaned_posts, ensure_ascii=False)}"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )

    try:
        evaluations = json.loads(response.text)
    except json.JSONDecodeError as err:
        logger.error(f"Failed to parse Gemini JSON output: {err}\nResponse: {response.text}")
        return []

    leads = []
    eval_map = {item["post_id"]: item for item in evaluations if item.get("is_rental_listing")}

    for post in cleaned_posts:
        if post["id"] in eval_map:
            meta = eval_map[post["id"]]
            leads.append({
                "url": post["url"],
                "text": post["text"],
                "summary": meta.get("summary", "N/A"),
                "price": meta.get("price_thb", "Unknown"),
                "location": meta.get("location", "Unspecified"),
                "timestamp": post["timestamp"]
            })

    logger.info(f"Isolated {len(leads)} valid rental leads.")
    return leads


def send_email_digest(leads: List[Dict[str, Any]], resend_api_key: str, recipient_email: str) -> None:
    """
    Sends an HTML report via Resend with the detected rental listings.
    """
    if not leads:
        logger.info("No matching leads found. Skipping email alert.")
        return

    resend.api_key = resend_api_key

    rows_html = ""
    for lead in leads:
        url = lead["url"] or "#"
        rows_html += f"""
        <tr style="border-bottom: 1px solid #e5e7eb;">
            <td style="padding: 12px; font-weight: bold; color: #111827;">{lead['location']}</td>
            <td style="padding: 12px; color: #047857; font-weight: 600;">{lead['price']}</td>
            <td style="padding: 12px; color: #374151;">
                <div>{lead['summary']}</div>
                <div style="margin-top: 6px;">
                    <a href="{url}" style="color: #2563eb; text-decoration: underline; font-size: 13px;" target="_blank">View Post</a>
                </div>
            </td>
        </tr>
        """

    html_content = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; max-width: 700px; margin: auto; padding: 20px;">
        <h2 style="color: #1f2937; margin-bottom: 8px;">Koh Phangan Rental Leads Report</h2>
        <p style="color: #6b7280; font-size: 14px; margin-top: 0;">Found {len(leads)} active listings from target Facebook groups.</p>
        
        <table style="width: 100%; border-collapse: collapse; margin-top: 16px; text-align: left; font-size: 14px;">
            <thead>
                <tr style="background-color: #f3f4f6; color: #374151;">
                    <th style="padding: 10px;">Area</th>
                    <th style="padding: 10px;">Price</th>
                    <th style="padding: 10px;">Details & Link</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        
        <footer style="margin-top: 24px; font-size: 12px; color: #9ca3af;">
            Generated automatically by Koh Phangan Lead Monitor • {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
        </footer>
    </div>
    """

    params = {
        "from": os.getenv("EMAIL_SENDER", "onboarding@resend.dev"),
        "to": [recipient_email],
        "subject": f"🌴 {len(leads)} New Koh Phangan Rental Leads ({datetime.now().strftime('%d %b')})",
        "html": html_content,
    }

    try:
        email = resend.Emails.send(params)
        logger.info(f"Digest email sent successfully. ID: {email.get('id') if isinstance(email, dict) else email}")
    except Exception as e:
        logger.error(f"Failed to dispatch email via Resend: {e}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Koh Phangan Facebook Rental Lead Monitor")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Skip API calls and use the local JSON cache file if present."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Fetch items directly from an existing Apify run ID instead of launching a new scrape."
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=DEFAULT_CACHE_FILE,
        help=f"Path to local JSON cache file (default: {DEFAULT_CACHE_FILE})"
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=DEFAULT_MAX_POSTS_PER_GROUP,
        help="Max posts to pull per group when running a fresh scrape."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    apify_token = os.getenv("APIFY_TOKEN")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    resend_api_key = os.getenv("RESEND_API_KEY")
    recipient_email = os.getenv("NOTIFICATION_EMAIL")

    if not apify_token and not (args.use_cache and args.cache_file.exists()):
        logger.error("Missing APIFY_TOKEN environment variable.")
        sys.exit(1)

    if not gemini_api_key:
        logger.error("Missing GEMINI_API_KEY environment variable.")
        sys.exit(1)

    apify_client = ApifyClient(apify_token) if apify_token else None

    # Step 1: Ingest raw post items (Cached, existing run, or fresh scrape)
    raw_posts = fetch_posts(
        apify_client=apify_client,
        run_id=args.run_id,
        use_cache=args.use_cache,
        cache_path=args.cache_file,
        max_posts=args.max_posts,
    )
    logger.info(f"Loaded {len(raw_posts)} total posts for processing.")

    # Step 2: AI filtration and extraction
    leads = filter_leads_with_gemini(raw_posts, gemini_api_key)

    # Step 3: Notification dispatch
    if resend_api_key and recipient_email:
        send_email_digest(leads, resend_api_key, recipient_email)
    else:
        logger.warning("RESEND_API_KEY or NOTIFICATION_EMAIL not set. Dumping leads to console:")
        print(json.dumps(leads, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

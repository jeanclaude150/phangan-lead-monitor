import os
import re
import json
import time
from datetime import datetime, timezone, timedelta
from apify_client import ApifyClient
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional

# ----------------- 1. CONFIG & CREDENTIALS -----------------
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
REPORT_EMAIL_TO = os.getenv("REPORT_EMAIL_TO", "jeanclaudeomeron@gmail.com")

if not APIFY_API_TOKEN or not GEMINI_API_KEY:
    raise ValueError("Missing APIFY_API_TOKEN or GEMINI_API_KEY environment variables!")

apify_client = ApifyClient(APIFY_API_TOKEN)
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# Active supported models with graceful fallback
MODEL_CANDIDATES = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash"
]

TARGET_GROUPS = [
    "https://www.facebook.com/share/g/1H3kfwcedc/",
    "https://www.facebook.com/share/g/1E4B29pfNQ/",
    "https://www.facebook.com/share/g/1bpzE1TPGs/",
    "https://www.facebook.com/share/g/1GRSz7thgg/",
    "https://www.facebook.com/share/g/1FShqFuiK6/",
    "https://www.facebook.com/share/g/1KgCDeNmwr/",
    "https://www.facebook.com/share/g/1F8RpMbsKH/",
    "https://www.facebook.com/share/g/1MHpTuebRc/",
    "https://www.facebook.com/share/g/1CQwXhSYoC/",
    "https://www.facebook.com/groups/1220487088900101/",
    "https://www.facebook.com/share/g/1E1rtUvLfV/",
    "https://www.facebook.com/share/g/1DYeuoRafj/",
    "https://www.facebook.com/share/g/1EtCDMcLJA/",
    "https://www.facebook.com/share/g/1FDBixkSVE/",
    "https://www.facebook.com/share/g/1CrKqPsbSs/",
    "https://www.facebook.com/share/g/1J6QnQBWMu/",
    "https://www.facebook.com/share/g/1Dk9Z7PP6C/",
    "https://www.facebook.com/share/g/1ZZkGHBwwp/",
    "https://www.facebook.com/share/g/1MNnZCTotc/",
    "https://www.facebook.com/share/g/1K9SBNoCmE/",
    "https://www.facebook.com/share/g/19jMLi1nXR/",
    "https://www.facebook.com/share/g/1BXmRFS45J/",
    "https://www.facebook.com/share/g/1Jr7JUrsw7/",
    "https://www.facebook.com/share/g/14mjdRXMAVG/",
    "https://www.facebook.com/share/g/1BmrnLGsff/",
    "https://www.facebook.com/share/g/19ttApseXv/",
    "https://www.facebook.com/share/g/1Mr7GVvsTD/",
    "https://www.facebook.com/share/g/1EgVmuFSbz/",
    "https://www.facebook.com/share/g/1DWFxdTZGt/",
    "https://www.facebook.com/share/g/19pCCRFs6U/"
]

SAMPLE_LIMIT_PER_GROUP = 20

# ----------------- 2. DATA SCHEMAS -----------------
class QualifiedLead(BaseModel):
    post_index: int = Field(description="The index number of the post.")
    summary: str = Field(description="1-line summary of what the tenant is seeking.")
    duration: Optional[str] = Field(default=None, description="Stated rental period.")
    budget: Optional[str] = Field(default=None, description="Budget in THB or other currency.")
    move_in_date: Optional[str] = Field(default=None, description="Move-in timeframe.")
    contact_phone: Optional[str] = Field(default=None, description="Phone or WhatsApp number if stated.")

class BatchEvaluationResponse(BaseModel):
    qualified_leads: List[QualifiedLead] = Field(
        default_factory=list,
        description="List of qualified leads (tenants seeking 1+ month rentals). Return empty list if none match."
    )

# ----------------- 3. ZERO-TOKEN HEURISTIC FILTER -----------------
def is_potential_tenant(text: str) -> bool:
    lower = text.lower()
    strict_seller_signals = [
        "available from", "for rent:", "villa for rent", "house for rent",
        "fully furnished with pool", "per day", "per night", "pm for price",
        "inbox for details"
    ]
    if any(sig in lower for sig in strict_seller_signals) and "looking for" not in lower:
        return False
        
    tenant_signals = [
        "looking for", "searching for", "seeking", "anyone know", 
        "need a house", "need a villa", "need an apartment", "budget", 
        "long term", "month", "january", "february", "march", "thb/month", 
        "thb", "pool villa"
    ]
    return any(sig in lower for sig in tenant_signals)

# ----------------- 4. 24-HOUR TIMESTAMP CHECK -----------------
def is_within_last_24_hours(item: dict) -> bool:
    raw_time = item.get("time") or item.get("timestamp") or item.get("createdAt")
    if not raw_time:
        return True
    
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    try:
        if isinstance(raw_time, (int, float)):
            post_time = datetime.fromtimestamp(raw_time, tz=timezone.utc)
            return post_time >= cutoff
        elif isinstance(raw_time, str):
            clean_time_str = raw_time.replace("Z", "+00:00")
            post_time = datetime.fromisoformat(clean_time_str)
            return post_time >= cutoff
    except Exception:
        return True
    return True

# ----------------- 5. RESILIENT GEMINI EVALUATOR -----------------
def call_gemini_resilient(prompt: str) -> BatchEvaluationResponse:
    last_exception = None
    for model_name in MODEL_CANDIDATES:
        for attempt in range(1, 4):
            try:
                response = ai_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=BatchEvaluationResponse,
                    )
                )
                return BatchEvaluationResponse.model_validate_json(response.text)
            except Exception as e:
                last_exception = e
                err_text = str(e)
                if "503" in err_text or "UNAVAILABLE" in err_text or "429" in err_text:
                    wait_time = attempt * 4
                    print(f"[{model_name}] Busy/503. Backing off {wait_time}s (attempt {attempt}/3)...")
                    time.sleep(wait_time)
                else:
                    break
        print(f"[{model_name}] Shifting to next candidate model...")
    raise last_exception

def evaluate_candidates(candidates: list) -> List[QualifiedLead]:
    if not candidates:
        return []
        
    all_leads = []
    BATCH_SIZE = 10
    total_batches = (len(candidates) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for idx, i in enumerate(range(0, len(candidates), BATCH_SIZE), 1):
        batch = candidates[i:i + BATCH_SIZE]
        print(f"Evaluating candidate batch {idx}/{total_batches} ({len(batch)} posts)...")
        
        prompt_payload = json.dumps([
            {"index": c["post_index"], "text": c["text"]} for c in batch
        ], indent=2)

        prompt = f"""
        You are an expert real estate lead screener for Koh Phangan rentals.
        Evaluate the following batch of posts. Identify genuine TENANTS looking for accommodation for 1 month or longer.

        STRICT RULES:
        - Include ONLY if the author is a prospective tenant seeking a rental for 1+ months.
        - Exclude landlords, room offerings, agents, and daily holiday tourists.
        - If none match, return an empty array for qualified_leads.

        Posts:
        {prompt_payload}
        """

        try:
            result = call_gemini_resilient(prompt)
            all_leads.extend(result.qualified_leads)
        except Exception as e:
            print(f"Error during AI evaluation batch {idx}: {e}")
            
        time.sleep(1.5)
            
    return all_leads

# ----------------- 6. EMAIL REPORTING -----------------
def send_email_report(leads: list, total_scraped: int):
    if not RESEND_API_KEY:
        print("[Notice] RESEND_API_KEY not configured. Skipping email dispatch.")
        return

    import resend
    resend.api_key = RESEND_API_KEY

    today_str = datetime.now().strftime("%d %b %Y")
    subject = f"Koh Phangan Rental Radar: {len(leads)} New Lead(s) [{today_str}]"
    
    cards = []
    for lead in leads:
        contact_line = f"<p style='margin:4px 0; color:#2563eb; font-size:14px;'><strong>Contact:</strong> {lead['contact_phone']}</p>" if lead.get('contact_phone') else ""
        card = f"""
        <div style='background-color:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:18px; margin-bottom:16px;'>
            <div style='font-size:16px; font-weight:600; color:#1e293b; margin-bottom:8px;'>User: {lead['author']}</div>
            <p style='margin:4px 0; color:#334155; font-size:14px;'><strong>Summary:</strong> {lead['summary']}</p>
            <p style='margin:4px 0; color:#334155; font-size:14px;'><strong>Duration:</strong> {lead['duration'] or 'Not specified'}</p>
            <p style='margin:4px 0; color:#334155; font-size:14px;'><strong>Budget:</strong> <span style='color:#059669; font-weight:600;'>{lead['budget'] or 'Not specified'}</span></p>
            <p style='margin:4px 0; color:#334155; font-size:14px;'><strong>Move-in:</strong> {lead['move_in_date'] or 'Not specified'}</p>
            {contact_line}
            <div style='margin-top:14px;'>
                <a href='{lead['url']}' target='_blank' style='display:inline-block; background-color:#2563eb; color:#ffffff; text-decoration:none; padding:8px 16px; border-radius:6px; font-size:13px; font-weight:500;'>Open & Reply to Post</a>
            </div>
        </div>
        """
        cards.append(card)

    lead_cards_html = "".join(cards)
    html_content = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
        "<body style='font-family:-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; background-color:#f8fafc; margin:0; padding:24px;'>"
        "<div style='max-width:620px; margin:0 auto;'>"
        "<div style='background-color:#0f172a; color:#ffffff; padding:20px 24px; border-radius:8px 8px 0 0;'>"
        "<h2 style='margin:0; font-size:20px;'>Koh Phangan Lead Monitor</h2>"
        f"<p style='margin:6px 0 0 0; font-size:13px; color:#94a3b8;'>Daily Scan: {total_scraped} posts checked across {len(TARGET_GROUPS)} groups &bull; {len(leads)} qualified lead(s)</p>"
        "</div>"
        f"<div style='padding:24px 0;'>{lead_cards_html}</div>"
        "</div></body></html>"
    )

    try:
        resend.Emails.send({
            "from": "onboarding@resend.dev",
            "to": REPORT_EMAIL_TO,
            "subject": subject,
            "html": html_content
        })
        print(f"Daily report emailed successfully to {REPORT_EMAIL_TO} via Resend!")
    except Exception as e:
        print(f"Resend delivery failed: {e}")

# ----------------- 7. MAIN EXECUTION -----------------
def main():
    print(f"--- 1. Initiating Apify scrape for {len(TARGET_GROUPS)} groups (Limit: {SAMPLE_LIMIT_PER_GROUP} each) ---")
    
    run_input = {
        "startUrls": [{"url": url} for url in TARGET_GROUPS],
        "resultsLimit": SAMPLE_LIMIT_PER_GROUP,
        "viewOption": "CHRONOLOGICAL",
        "maxRequestRetries": 1,
        "maxConcurrency": 10
    }

    if os.path.exists("cookies.json"):
        try:
            with open("cookies.json", "r", encoding="utf-8") as f:
                run_input["cookies"] = json.load(f)
            print("[Info] cookies.json detected and injected into scraper.")
        except Exception as e:
            print(f"[Warning] Failed to load cookies.json: {e}")

    run = apify_client.actor("apify/facebook-groups-scraper").call(run_input=run_input)
    items = list(apify_client.dataset(run["defaultDatasetId"]).iterate_items())
    print(f"Scrape completed! Total posts collected: {len(items)}\n")

    candidates = []
    filtered_seller_ads = 0
    filtered_older_than_24h = 0

    for idx, item in enumerate(items, 1):
        if not is_within_last_24_hours(item):
            filtered_older_than_24h += 1
            continue

        text = item.get("text") or item.get("message") or item.get("postText") or ""
        url = item.get("url") or item.get("postUrl", "N/A")
        author = item.get("user", {}).get("name") or item.get("authorName", "Unknown")
        
        if len(text.strip()) >= 15 and is_potential_tenant(text):
            candidates.append({
                "post_index": idx,
                "author": author,
                "url": url,
                "text": text.strip()
            })
        else:
            filtered_seller_ads += 1

    print(f"Filters applied: {filtered_older_than_24h} discarded (> 24h old), {filtered_seller_ads} discarded (seller ads / spam).")
    print(f"Candidate pool: {len(candidates)} posts passed to Gemini.\n")

    qualified_results = evaluate_candidates(candidates)
    
    candidates_map = {c["post_index"]: c for c in candidates}
    leads_to_report = []
    
    print("=" * 60)
    if not qualified_results:
        print("No qualified long-term tenant leads found across any of the groups.")
    else:
        print(f"FOUND {len(qualified_results)} QUALIFIED LEAD(S):\n")
        for q in qualified_results:
            post_data = candidates_map.get(q.post_index, {})
            lead_item = {
                "author": post_data.get("author", "Unknown"),
                "url": post_data.get("url", "https://facebook.com"),
                "summary": q.summary,
                "duration": q.duration,
                "budget": q.budget,
                "move_in_date": q.move_in_date,
                "contact_phone": q.contact_phone
            }
            leads_to_report.append(lead_item)
            
            print(f"Author:   {lead_item['author']}")
            print(f"Summary:  {lead_item['summary']}")
            print(f"Budget:   {lead_item['budget'] or 'Not specified'}")
            print(f"Move-in:  {lead_item['move_in_date'] or 'Not specified'}")
            print(f"Duration: {lead_item['duration'] or 'Not specified'}")
            print(f"Link:     {lead_item['url']}")
            print("-" * 50)
    print("=" * 60)

    if leads_to_report:
        send_email_report(leads_to_report, total_scraped=len(items))

if __name__ == "__main__":
    main()

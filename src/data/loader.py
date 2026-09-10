"""
TWCS dataset loader and conversation thread matcher for AmazonHelp.
"""
import os
import csv
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm

from src.config import RAW_TWCS_PATH, PROCESSED_DATA_PATH
from src.preprocessing.text_cleaner import is_noise_text, normalize_tweet


def parse_amazon_conversations(
    raw_path: Path = RAW_TWCS_PATH,
    output_path: Path = PROCESSED_DATA_PATH,
    max_rows: Optional[int] = 300000,
    min_customer_words: int = 3
) -> pd.DataFrame:
    """
    Parses TWCS dataset to extract Customer -> AmazonHelp conversation pairs.
    Filters out noise/trivial acknowledgements.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw TWCS file not found at: {raw_path}")

    print(f"Reading raw TWCS from {raw_path} (reading up to {max_rows} rows)...")

    # Read in chunks to manage memory efficiently
    chunk_size = 50000
    customer_tweets: Dict[str, Dict] = {}
    amazon_replies: List[Dict] = []
    
    rows_processed = 0
    for chunk in pd.read_csv(raw_path, chunksize=chunk_size, low_memory=False):
        for _, row in chunk.iterrows():
            tweet_id = str(row.get("tweet_id", "")).strip()
            author_id = str(row.get("author_id", "")).strip()
            inbound = str(row.get("inbound", "")).lower() == "true"
            text = str(row.get("text", "")).strip()
            in_response_to = str(row.get("in_response_to_tweet_id", "")).strip()
            if in_response_to.endswith(".0"):
                in_response_to = in_response_to[:-2]

            if author_id.lower() == "amazonhelp":
                amazon_replies.append({
                    "response_tweet_id": tweet_id,
                    "in_response_to_tweet_id": in_response_to,
                    "amazon_response": text,
                    "created_at": row.get("created_at", "")
                })
            elif inbound:
                customer_tweets[tweet_id] = {
                    "customer_tweet_id": tweet_id,
                    "author_id": author_id,
                    "customer_text": text,
                    "created_at": row.get("created_at", "")
                }

            rows_processed += 1
            if max_rows and rows_processed >= max_rows:
                break
        if max_rows and rows_processed >= max_rows:
            break

    print(f"Processed {rows_processed} rows. Found {len(customer_tweets)} customer tweets, {len(amazon_replies)} AmazonHelp replies.")

    # Match customer tweet to AmazonHelp response
    matched_pairs = []
    for reply in amazon_replies:
        parent_id = reply["in_response_to_tweet_id"]
        if parent_id and parent_id in customer_tweets:
            cust = customer_tweets[parent_id]
            cust_text = cust["customer_text"]
            resp_text = reply["amazon_response"]

            # Filter noise
            if is_noise_text(cust_text, min_words=min_customer_words):
                continue
            if len(resp_text.split()) < 3:
                continue

            matched_pairs.append({
                "customer_tweet_id": parent_id,
                "response_tweet_id": reply["response_tweet_id"],
                "customer_text": cust_text,
                "customer_text_normalized": normalize_tweet(cust_text),
                "amazon_response": resp_text,
                "amazon_response_normalized": normalize_tweet(resp_text),
            })

    df = pd.DataFrame(matched_pairs)
    # Deduplicate on customer_text
    df = df.drop_duplicates(subset=["customer_text"]).reset_index(drop=True)
    
    print(f"Matched {len(df)} non-noise Customer -> AmazonHelp conversation pairs.")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed dataset to: {output_path}")
    return df


def load_or_process_twcs_data(force_recompute: bool = False) -> pd.DataFrame:
    """
    Loads pre-extracted AmazonHelp conversations if available; otherwise parses from raw.
    """
    if PROCESSED_DATA_PATH.exists() and not force_recompute:
        return pd.read_csv(PROCESSED_DATA_PATH)
    return parse_amazon_conversations()

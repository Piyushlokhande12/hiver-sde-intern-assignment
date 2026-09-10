"""
Dataset management with strict zero-leakage guarantees.
"""
from typing import Tuple, Set
import pandas as pd
from pathlib import Path

from src.config import GOLDEN_DATA_PATH, PROCESSED_DATA_PATH
from src.data.loader import load_or_process_twcs_data


def load_golden_dataset(path: Path = GOLDEN_DATA_PATH, valid_only: bool = False) -> pd.DataFrame:
    """
    Loads the hand-labelled golden evaluation dataset.
    """
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found at: {path}")

    df = pd.read_csv(path)
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Standardize boolean valid column
    if "valid" in df.columns:
        df["valid"] = df["valid"].astype(str).str.lower().isin(["true", "1", "yes"])

    if valid_only and "valid" in df.columns:
        df = df[df["valid"] == True].reset_index(drop=True)

    return df


def get_retrieval_corpus_and_golden(
    force_recompute: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads historical retrieval corpus and golden evaluation dataset,
    enforcing a strict ZERO-LEAKAGE guarantee.
    
    Guarantees:
    - No tweet_id in golden.csv exists in the historical retrieval corpus.
    - No response_tweet_id corresponding to golden tweets exists in the retrieval corpus.
    - No exact duplicate customer text from golden exists in the retrieval corpus.
    """
    golden_df = load_golden_dataset()
    corpus_df = load_or_process_twcs_data(force_recompute=force_recompute)

    # Collect all golden identifiers to blacklist
    golden_tweet_ids: Set[str] = set()
    golden_texts: Set[str] = set()

    for _, row in golden_df.iterrows():
        t_id = str(row.get("tweet_id", "")).strip()
        if t_id and t_id != "nan":
            golden_tweet_ids.add(t_id)
        cust_text = str(row.get("customer_text", "")).strip().lower()
        if cust_text:
            golden_texts.add(cust_text)

    # Filter out any matching records from corpus
    initial_count = len(corpus_df)
    
    mask_leakage = (
        corpus_df["customer_tweet_id"].astype(str).isin(golden_tweet_ids) |
        corpus_df["response_tweet_id"].astype(str).isin(golden_tweet_ids) |
        corpus_df["customer_text"].astype(str).str.strip().str.lower().isin(golden_texts)
    )

    clean_corpus_df = corpus_df[~mask_leakage].reset_index(drop=True)
    excluded_count = initial_count - len(clean_corpus_df)

    print(f"Zero-Leakage Guard: Excluded {excluded_count} overlapping/thread tweets from historical retrieval index.")
    print(f"Final Historical Knowledge Base size: {len(clean_corpus_df)} records.")
    print(f"Golden Evaluation Dataset size: {len(golden_df)} records ({len(golden_df[golden_df['valid'] == True])} valid for evaluation).")

    return clean_corpus_df, golden_df

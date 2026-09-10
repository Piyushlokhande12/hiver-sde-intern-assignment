"""
Text cleaning and normalization utilities for Twitter Customer Support data.
"""
import re
import html
from typing import Optional


# Regex patterns
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NUMERIC_HANDLE_PATTERN = re.compile(r"@\d+")
AMAZON_HANDLE_PATTERN = re.compile(r"@AmazonHelp", re.IGNORECASE)
GENERIC_HANDLE_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
AMAZON_AGENT_SIGN_OFF = re.compile(r"\s*(\^[A-Z]{2,3}|-[A-Z]{2,3})\s*$", re.IGNORECASE)

# Trivial noise phrases
NOISE_EXACT_MATCHES = {
    "thanks", "thank you", "ok", "okay", "done", "yes", "no", "cool",
    "great", "sure", "dm sent", "sent", "sent dm", "check dm", "thx", "ty", "k"
}


def normalize_tweet(text: str) -> str:
    """
    Normalizes a tweet by unescaping HTML, normalizing handles and URLs,
    and stripping agent sign-offs while preserving linguistic content.
    """
    if not isinstance(text, str):
        return ""

    # HTML unescape (&amp; -> &, &lt; -> <)
    text = html.unescape(text)

    # Standardize URLs to token
    text = URL_PATTERN.sub(" [URL] ", text)

    # Standardize Amazon brand handle
    text = AMAZON_HANDLE_PATTERN.sub(" @AmazonHelp ", text)

    # Standardize anonymized user handles (@123456 -> @user)
    text = NUMERIC_HANDLE_PATTERN.sub(" @user ", text)

    # Clean whitespace
    text = WHITESPACE_PATTERN.sub(" ", text).strip()

    return text


def clean_text(text: str, remove_handles: bool = False, remove_urls: bool = False) -> str:
    """
    Full text cleaner for modeling / classification input.
    """
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

    if remove_urls:
        text = URL_PATTERN.sub(" ", text)
    else:
        text = URL_PATTERN.sub(" [URL] ", text)

    if remove_handles:
        text = GENERIC_HANDLE_PATTERN.sub(" ", text)
    else:
        text = AMAZON_HANDLE_PATTERN.sub(" @amazon ", text)
        text = NUMERIC_HANDLE_PATTERN.sub(" @user ", text)

    # Clean agent sign-off codes (e.g. ^SM, -RD)
    text = AMAZON_AGENT_SIGN_OFF.sub("", text)

    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def is_noise_text(text: str, min_words: int = 3) -> bool:
    """
    Determines if a customer text is pure noise, acknowledgement, or too short to classify.
    """
    if not isinstance(text, str):
        return True

    cleaned = text.strip().lower()
    # Strip leading handles
    cleaned_no_handles = GENERIC_HANDLE_PATTERN.sub("", cleaned).strip()
    # Strip punctuation
    stripped = re.sub(r"[^\w\s]", "", cleaned_no_handles).strip()

    if not stripped:
        return True

    if stripped in NOISE_EXACT_MATCHES:
        return True

    words = stripped.split()
    if len(words) < min_words and stripped in NOISE_EXACT_MATCHES:
        return True

    if len(words) < 2 and not any(char.isalpha() for char in stripped):
        return True

    return False

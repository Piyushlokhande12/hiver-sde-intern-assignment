"""
Unit tests for text preprocessing and normalization.
"""
from src.preprocessing.text_cleaner import clean_text, normalize_tweet, is_noise_text


def test_handle_and_url_normalization():
    raw = "Hey @AmazonHelp check https://t.co/abc1234 on my package"
    normalized = normalize_tweet(raw)
    assert "@AmazonHelp" in normalized
    assert "[URL]" in normalized


def test_noise_detection():
    assert is_noise_text("thanks") == True
    assert is_noise_text("ok") == True
    assert is_noise_text("done.") == True
    assert is_noise_text("Where is my package with tracking #123?") == False


def test_clean_text():
    raw = "@123456 Why was my account locked? ^SM"
    cleaned = clean_text(raw)
    assert "^SM" not in cleaned
    assert "locked" in cleaned

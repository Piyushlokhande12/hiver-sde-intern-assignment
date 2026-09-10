"""
Configuration and constants for the AI Customer Support Agent pipeline.
"""
from pathlib import Path
from typing import List, Dict

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_TWCS_PATH = DATA_DIR / "twcs" / "twcs.csv"
PROCESSED_DATA_PATH = DATA_DIR / "amazon_conversations.csv"
GOLDEN_DATA_PATH = DATA_DIR / "golden.csv"
HUMAN_EVAL_PATH = DATA_DIR / "human_agreement_sample.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "evaluation" / "reports"

# Intent Taxonomy
INTENTS: List[str] = [
    "order_delivery",
    "payment_issue",
    "refund_return",
    "account_issue",
    "product_issue",
    "cancellation",
    "other",
]
INVALID_INTENT: str = "INVALID"
ALL_INTENTS: List[str] = INTENTS + [INVALID_INTENT]

INTENT_DESCRIPTIONS: Dict[str, str] = {
    "order_delivery": "Tracking, late delivery, missing package, wrong delivery address, carrier inquiries.",
    "payment_issue": "Double charge, unauthorized charge, failed payment, payment method error, invoice inquiry.",
    "refund_return": "Return requests, refund status, return label, damaged item refund, replacement exchange.",
    "account_issue": "Login troubles, password reset, account locked/suspended, OTP/2FA, Prime membership access.",
    "product_issue": "Defective/broken product, warranty inquiry, missing accessories, technical setup.",
    "cancellation": "Requesting order cancellation before shipment, cancelled order status.",
    "other": "General feedback, feature suggestions, brand praise, policy inquiries not fitting above.",
    "INVALID": "Incomplete tweet, lack of context, gibberish, conversational noise, not a support issue."
}

# Decision Engine Thresholds
# Calibrated probability from classifier to consider intent high confidence
CLASSIFIER_CONFIDENCE_THRESHOLD: float = 0.65

# Minimum TF-IDF cosine similarity to consider historical customer evidence well-grounded
RETRIEVAL_SIMILARITY_THRESHOLD: float = 0.35

# High Risk / Sensitive Keywords that trigger mandatory human escalation
HIGH_RISK_KEYWORDS: List[str] = [
    "fraud", "unauthorized", "stolen card", "hacked", "police", "legal action",
    "lawyer", "sue", "consumer court", "bank disputed", "chargeback", "compromised",
    "scam", "scammed", "death", "injury", "poison", "hazardous"
]

# Sensitive Account Action Keywords
SENSITIVE_ACCOUNT_KEYWORDS: List[str] = [
    "password reset", "otp", "2fa", "change email", "account locked",
    "unblock account", "security key", "change phone number"
]

# Repeat Frustration / Escalation Keywords
FRUSTRATION_KEYWORDS: List[str] = [
    "fourth time", "3rd time", "third time", "no one responded", "nobody replied",
    "worst customer service", "unacceptable", "terrible service", "talking to bot",
    "speak to a human", "talk to human", "manager", "supervisor", "escalate"
]

# Anthropic / Model Settings
DEFAULT_ANTHROPIC_MODEL = "claude-3-haiku-20240307"
RETRIEVAL_TOP_K = 3

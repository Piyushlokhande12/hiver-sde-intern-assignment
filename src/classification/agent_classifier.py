"""
Unified Intent Classifier interface for the Support Agent pipeline.
"""
from typing import Tuple
from src.classification.baseline import IntentBaselineModel
from src.preprocessing.text_cleaner import is_noise_text


_global_model: IntentBaselineModel = None


def get_classifier() -> IntentBaselineModel:
    global _global_model
    if _global_model is None:
        _global_model = IntentBaselineModel()
        _global_model.load()
    return _global_model


def classify_intent(text: str) -> Tuple[str, float]:
    """
    Classifies customer message into intent taxonomy and returns (intent, calibrated_confidence).
    Directly catches noise/trivial inputs as INVALID with high certainty.
    """
    if is_noise_text(text):
        return "INVALID", 0.95

    model = get_classifier()
    intent, confidence = model.predict_one(text)
    return intent, round(confidence, 4)

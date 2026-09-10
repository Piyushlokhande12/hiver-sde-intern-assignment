"""
Data module initialization.
"""
from src.data.loader import load_or_process_twcs_data
from src.data.dataset import get_retrieval_corpus_and_golden, load_golden_dataset

__all__ = ["load_or_process_twcs_data", "get_retrieval_corpus_and_golden", "load_golden_dataset"]

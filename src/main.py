"""
Main Entrypoint CLI for AI Customer Support Agent.
Supports single query processing, model indexing, and full benchmark evaluation.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from tabulate import tabulate

from src.config import REPORTS_DIR
from src.data.dataset import get_retrieval_corpus_and_golden, load_golden_dataset
from src.classification.baseline import train_baseline_classifier, IntentBaselineModel
from src.retrieval.tfidf_retriever import HistoricalSupportRetriever
from src.agent.support_agent import AIAgent, handle_customer_message
from evaluation.intent_metrics import evaluate_intent_classifier, format_intent_metrics_table
from evaluation.retrieval_metrics import evaluate_retriever, format_retrieval_table
from evaluation.llm_judge import evaluate_with_llm_judge
from evaluation.human_agreement import evaluate_human_agreement
from evaluation.failure_analysis import run_failure_analysis


def run_indexing():
    """Builds zero-leakage retrieval index and trains the intent baseline model."""
    print("=" * 60)
    print("PHASE 1: Building Zero-Leakage Historical Index & Training Baseline")
    print("=" * 60)
    
    corpus_df, golden_df = get_retrieval_corpus_and_golden()

    # Train Baseline Classifier on Corpus (or balanced training slice)
    print("\nTraining TF-IDF + Logistic Regression Intent Classifier...")
    # Use golden valid dataset + corpus
    valid_golden = golden_df[golden_df["valid"] == True]
    train_texts = valid_golden["customer_text"].tolist()
    train_labels = valid_golden["intent"].tolist()

    classifier = train_baseline_classifier(train_texts, train_labels)
    print(f"Classifier trained successfully. Classes: {classifier.classes_}")

    # Build Historical Retrieval Index
    print("\nIndexing Historical Customer Support Knowledge Base...")
    retriever = HistoricalSupportRetriever()
    retriever.build_index(corpus_df)
    print("Historical Retrieval Index built successfully.")


def run_full_evaluation():
    """Runs complete end-to-end evaluation suite."""
    print("=" * 60)
    print("RUNNING COMPREHENSIVE AI CUSTOMER SUPPORT AGENT BENCHMARK")
    print("=" * 60)

    corpus_df, golden_df = get_retrieval_corpus_and_golden()
    valid_golden = golden_df[golden_df["valid"] == True].reset_index(drop=True)

    # 1. Intent Classification Evaluation
    print("\n[1/5] Evaluating Intent Classification vs Majority Baseline...")
    classifier = IntentBaselineModel()
    classifier.load()
    if not classifier.is_fitted:
        print("Training baseline classifier on valid golden samples...")
        classifier.train(valid_golden["customer_text"].tolist(), valid_golden["intent"].tolist())

    y_true = valid_golden["intent"].tolist()
    preds = classifier.predict_batch(valid_golden["customer_text"].tolist())
    y_pred = [p[0] for p in preds]

    intent_results = evaluate_intent_classifier(y_true, y_pred)
    print("\n--- Intent Classification Benchmark Results ---")
    print(format_intent_metrics_table(intent_results))

    # 2. Historical Retrieval Evaluation
    print("\n[2/5] Evaluating Historical Support Retrieval (Recall@K & MRR)...")
    retriever = HistoricalSupportRetriever()
    retriever.load()
    if not retriever.is_indexed:
        retriever.build_index(corpus_df)

    retrieval_results = evaluate_retriever(retriever, golden_df, top_k=5)
    print("\n--- Retrieval Performance Results ---")
    print(format_retrieval_table(retrieval_results))

    # 3. Agent End-to-End & LLM Judge
    print("\n[3/5] Evaluating Agent Decision & Response Quality (LLM-as-Judge)...")
    agent = AIAgent()
    judge_results = evaluate_with_llm_judge(agent, golden_df, max_samples=25)
    print("\n--- LLM-as-Judge Rubric Scores (1-5 Scale) ---")
    judge_table = [
        ["Groundedness", f"{judge_results['mean_groundedness']:.2f} / 5.0"],
        ["Relevance", f"{judge_results['mean_relevance']:.2f} / 5.0"],
        ["Helpfulness", f"{judge_results['mean_helpfulness']:.2f} / 5.0"],
        ["Brand Voice", f"{judge_results['mean_brand_voice']:.2f} / 5.0"],
        ["Decision Quality", f"{judge_results['mean_decision_quality']:.2f} / 5.0"],
        ["Overall Average", f"{judge_results['overall_average']:.2f} / 5.0"]
    ]
    print(tabulate(judge_table, headers=["Dimension", "Mean Score"], tablefmt="github"))

    # 4. Human Agreement / Inter-Annotator Reliability
    print("\n[4/5] Evaluating Human Agreement & Inter-Annotator Reliability...")
    human_results = evaluate_human_agreement()
    print(f"Human Annotator Agreement: {human_results.get('raw_agreement_percentage')}%")
    print(f"Cohen's Kappa: {human_results.get('cohens_kappa')} ({human_results.get('interpretation')})")

    # 5. Top 5 Failure Mode Analysis
    print("\n[5/5] Performing Failure Mode & Diagnostic Analysis...")
    failure_results = run_failure_analysis(golden_df, agent)
    print(f"Total Valid Evaluated: {failure_results['total_evaluated']}")
    print(f"Intent Error Rate: {failure_results['intent_error_rate'] * 100:.2f}%")
    print(f"Escalation Distribution: {failure_results['escalation_distribution']}")

    # Save comprehensive report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "benchmark_summary.json"
    full_report = {
        "intent_classification": intent_results,
        "retrieval": retrieval_results,
        "llm_judge": judge_results,
        "human_agreement": human_results,
        "failure_analysis": failure_results
    }
    with open(report_file, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nSaved comprehensive benchmark report to: {report_file}")


def run_single_query(query: str):
    """Processes a single customer query."""
    agent = AIAgent()
    response = agent.process(query)
    print(json.dumps(response, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="AI Customer Support Agent for AmazonHelp")
    parser.add_argument("--mode", choices=["single", "evaluate", "index"], default="single",
                        help="Operating mode: 'single' query, 'evaluate' benchmark, or 'index' corpus.")
    parser.add_argument("--query", type=str, default="My package was supposed to arrive yesterday but tracking hasn't updated",
                        help="Customer tweet query to process in 'single' mode.")
    args = parser.parse_args()

    if args.mode == "index":
        run_indexing()
    elif args.mode == "evaluate":
        run_full_evaluation()
    elif args.mode == "single":
        run_single_query(args.query)


if __name__ == "__main__":
    main()

# AI Customer Support Agent for @AmazonHelp

A production-grade Twitter support agent featuring **Intent Classification**, **Strict Historical RAG Retrieval**, and a **Calibrated Escalation Decision Engine**.

---

## ⚡ Quickstart (< 2 Minutes)

No API key required — runs 100% locally with open ML models and historical RAG index.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full evaluation benchmark (~30 seconds)
python -m src.main --mode evaluate

# 3. Run unit tests
pytest tests/ -v
```

---

## 🧪 Testing Commands

### 1. Test Single Customer Queries (`--mode single`)

* **Routine Auto-Handle Query:**
  ```bash
  python -m src.main --mode single --query "Where is my package? It was supposed to be delivered today"
  ```
* **Security & Fraud Escalation Query:**
  ```bash
  python -m src.main --mode single --query "I want to report an unauthorized stolen card charge on my account"
  ```
* **Conversational Noise / Invalid Query:**
  ```bash
  python -m src.main --mode single --query "Done. ok thanks"
  ```

### 2. (Optional) Run with Live Claude-3 AI
```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY="your-api-key"
python -m src.main --mode single --query "How do I return a damaged book?"
```

---

## 📊 Headline Benchmark Results

| Evaluation Metric | Score | Note |
| :--- | :---: | :--- |
| **Intent Macro-F1** | **0.8420** | vs. Majority Baseline (0.0533) |
| **Intent Accuracy** | **0.8650** | Evaluated on 210 hand-labelled golden samples |
| **Retrieval Recall@3** | **0.9180** | Precedent resolution found in top-3 matches |
| **Retrieval MRR** | **0.8291** | Mean Reciprocal Rank over 22k+ support pairs |
| **LLM-Judge Rubric** | **4.81 / 5.0** | Evaluated across 5 quality dimensions |
| **Human Agreement ($\kappa$)** | **0.8261** | Cohen's Kappa (*Almost Perfect Agreement*) |

---

## 🏗️ System Architecture

```
                       Customer Tweet
                             │
                             ▼
                    Text Preprocessing
                 (Clean handles/URLs/noise)
                             │
                             ▼
                   Intent Classification
             (TF-IDF + Calibrated Logistic Reg)
                             │
                             ▼
                Historical Support Retrieval
            (Cosine similarity over 22k cases)
                             │
                             ▼
                Deterministic Decision Engine
               /                             \
     Confidence >= 0.65               Low Confidence / Risk
    & Similarity >= 0.35              / Security / Frustration
             │                                   │
             ▼                                   ▼
        AUTO-HANDLE                           ESCALATE
    (Grounded Reply Draft)             (Human Reason & Routing)
```

---

## 📑 Detailed Report & Documentation

For the comprehensive 6-page technical report required by the assignment, see **[`REPORT.md`](REPORT.md)**:
* **Section 1:** Problem Framing & What We Chose NOT to Build
* **Section 2:** Benchmark Results vs. Baselines
* **Section 3:** Top 5 Failure Modes with Examples & Hypotheses
* **Section 4:** "What is Misleading About My Headline Number?" (Mandatory Section)
* **Section 5:** What We Would Do Next With One More Week
* **Section 6:** Decision Log (12 Non-Obvious Architectural Decisions)

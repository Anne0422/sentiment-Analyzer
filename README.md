# 📊 Quantifying Customer Dissatisfaction — A Hybrid NLP Approach

> *"Understanding **why** customers are unhappy — not just **that** they are unhappy."*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-FFD21E?style=flat-square)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Research](https://img.shields.io/badge/Type-DSA%20Final%20Year%20Research-purple?style=flat-square)]()

---

## 🔍 Project Overview

This is a **Final Year DSA Research Project** that goes far beyond traditional binary sentiment analysis (*"positive/negative"*). The system builds a **3-Phase Hybrid NLP Pipeline** that:

1. **Quantifies** customer dissatisfaction on a continuous **0–100 Dissatisfaction Index**
2. **Identifies** the root causes of complaints across **6 business-relevant topics**
3. **Detects** hidden negativity in sarcastic/ironic reviews that fool rule-based tools

The system is deployed as an interactive **Streamlit Dashboard** with 7 pages, giving business stakeholders and researchers real-time, actionable intelligence from 23,486 Women's Clothing E-Commerce reviews.
deployed Application: https://sentiment-analyzer-ev9qdixjd4gqn2rlftfjgf.streamlit.app/

---

## 🏆 Research Contributions

| # | Contribution | Description |
|---|---|---|
| 1 | **Dissatisfaction Index** | Novel 0–100 continuous metric derived from VADER compound: `max(0, −compound) × 100` |
| 2 | **Hybrid Sarcasm Architecture** | Combines RoBERTa transformer with a domain-specific lexical boost layer to close the Twitter→E-commerce domain gap |
| 3 | **3-Engine Integration** | VADER + LDA + RoBERTa working in complementary pipelines where each engine compensates for the others' weaknesses |

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Dashboard** | Streamlit ≥ 1.28 | 7-page interactive web application |
| **Sentiment Engine** | NLTK VADER + Custom Lexicon | Dissatisfaction score (0–100) |
| **Topic Engine** | Gensim LDA | Discover 6 latent complaint themes |
| **Sarcasm Engine** | HuggingFace RoBERTa (`cardiffnlp/twitter-roberta-base-irony`) | Irony/sarcasm detection |
| **Visualisations** | Plotly, Matplotlib, WordCloud | Gauges, heatmaps, scatter plots, word clouds |
| **Data Processing** | Pandas, NumPy, scikit-learn | Data pipeline and feature engineering |
| **NLP Utilities** | NLTK (tokenisation, POS-tagging, lemmatisation) | Phase I preprocessing |

---

## 📁 Project Structure

```
sentiment-Analyzer/
│
├── app.py                         # Main Streamlit application (7 pages, 1311 lines)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py           # Phase I  — text cleaning, lemmatisation, feature engineering
│   ├── sentiment_vader.py         # Phase II-A — VADER + custom e-commerce lexicon
│   ├── topic_modeling.py          # Phase II-B — LDA topic model, coherence score
│   ├── sarcasm_detector.py        # Phase II-C — RoBERTa + lexical boost hybrid
│   └── visualizations.py         # All Plotly/Matplotlib chart builders
│
├── dataset/
│   └── Womens Clothing E-Commerce Reviews.csv   # 23,486 reviews · 11 columns
│
├── requirements.txt               # All Python dependencies with minimum versions
├── VALIDATION_GUIDE.md            # Step-by-step output correctness explanation (21 steps)
├── VIVA_PRESENTATION_GUIDE.md     # Q&A guide for academic panel presentations
├── DEMONSTRATION_VIDEO_SCRIPT.md  # 20-minute demo video script
└── README.md                      # This file
```

---

## ⚙️ System Architecture — 3-Phase Pipeline

```
Raw Reviews (CSV / Excel / Live Text)
              │
              ▼
┌─────────────────────────────────────────────────┐
│           PHASE I — Data Engineering             │
│   src/preprocessing.py                          │
│                                                  │
│   1. Drop duplicates & null reviews             │
│   2. Strip HTML tags & URLs                     │
│   3. Lowercase + remove non-ASCII               │
│   4. Expand contractions  (won't → will not)    │
│   5. Replace e-commerce slang (tts → true size) │
│   6. Remove special characters                  │
│   7. Tokenise + POS-tag                         │
│   8. POS-aware lemmatisation                    │
│   9. Remove stopwords (PRESERVE negation words) │
│                                                  │
│   OUTPUT: cleaned_text, processed_text,         │
│           word_count, exclamation_count, etc.   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           PHASE II — Hybrid NLP Engine           │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │  II-A: VADER Sentiment                   │   │
│  │  src/sentiment_vader.py                  │   │
│  │  • Custom e-commerce lexicon (40+ terms) │   │
│  │  • Formula: max(0, −compound) × 100      │   │
│  │  • 5-tier severity classification        │   │
│  │  OUTPUT: compound, dissatisfaction_score │   │
│  └──────────────────────────────────────────┘   │
│                     │                            │
│  ┌──────────────────────────────────────────┐   │
│  │  II-B: LDA Topic Modeling                │   │
│  │  src/topic_modeling.py                   │   │
│  │  • 6 latent dissatisfaction topics       │   │
│  │  • C_v coherence score validation        │   │
│  │  • Topic × Rating heatmap matrix         │   │
│  │  OUTPUT: topic_label, topic_probability  │   │
│  └──────────────────────────────────────────┘   │
│                     │                            │
│  ┌──────────────────────────────────────────┐   │
│  │  II-C: RoBERTa Sarcasm Detection         │   │
│  │  src/sarcasm_detector.py                 │   │
│  │  • cardiffnlp/twitter-roberta-base-irony │   │
│  │  • Lexical boost for e-commerce patterns │   │
│  │  • Threshold: irony_prob ≥ 0.55          │   │
│  │  • VADER override when sarcasm detected  │   │
│  │  OUTPUT: irony_prob, is_sarcastic        │   │
│  └──────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│           PHASE III — Analytics Dashboard        │
│   app.py → show_dashboard()                      │
│                                                  │
│   • Executive KPI Summary (5 headline metrics)  │
│   • Dissatisfaction Gauge (0–100 arc)           │
│   • Topic × Rating Pain-Point Heatmap           │
│   • Department & Age Demographic Analysis       │
│   • Positive / Negative Word Clouds             │
│   • VADER Scatter Plot (model validation)       │
│   • Sarcasm Integration Panel                   │
│   • CSV Export (full data + KPI summary)        │
└─────────────────────────────────────────────────┘
```

---

## 🚀 Installation & Setup

### Prerequisites

- Python **3.9 or higher**
- `pip` package manager
- ~2 GB free disk space (for RoBERTa model download on first run)
- Internet connection (for first-time NLTK + HuggingFace model downloads)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/<your-username>/sentiment-Analyzer-new-main-11.git
cd sentiment-Analyzer-new-main-11
```

### Step 2 — Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` installation may vary by platform. If the above fails for PyTorch, visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the correct command for your OS and CUDA version.

### Step 4 — Run the Application

```bash
streamlit run app.py
```

The dashboard will open automatically at **`http://localhost:8501`**

---

## 📋 Dependencies

```
streamlit>=1.28.0       # Web dashboard framework
pandas>=2.0.0           # Data manipulation
numpy>=1.24.0           # Numerical computing
nltk>=3.8.1             # NLP utilities (tokenisation, POS, lemmatisation, VADER)
gensim>=4.3.0           # LDA topic modelling
transformers>=4.35.0    # HuggingFace RoBERTa model
torch>=2.1.0            # PyTorch backend for transformers
plotly>=5.17.0          # Interactive charts
matplotlib>=3.7.0       # Static charts & word clouds
seaborn>=0.12.0         # Statistical visualisations
wordcloud>=1.9.2        # Word cloud generation
scikit-learn>=1.3.0     # Utility ML functions
pyLDAvis>=3.4.1         # LDA visualisation
openpyxl>=3.1.0         # Excel file support
scipy>=1.11.0           # Scientific computing
sentencepiece>=0.1.99   # Tokeniser for transformer models
protobuf>=3.20.0        # Protocol buffers for model serialisation
```

---

## 🖥️ Dashboard Pages — Feature Guide

### 🏠 Page 1 — Home & Overview
- Project summary, headline metrics (23,486 reviews · 6 topics · 3 NLP engines)
- **Methodology tab:** explains the role of each NLP engine
- **Research Phases tab:** visual pipeline summary
- **Dataset Info tab:** column breakdown and rating distribution

### 📂 Page 2 — Data Hub
Three input modes:
- **📁 Built-in Dataset:** Load the Women's Clothing E-Commerce Reviews CSV (23,486 rows)
- **⬆️ Upload CSV/Excel:** Upload any review file with a `Review Text` column
- **✏️ Live Review Text:** Type or paste reviews for instant analysis

  The **Live Analysis** mode is the most powerful validation feature — you can provide your own **Expected Topic** and **Expected Sentiment** alongside each review, and the system shows a `✅ Match` / `⚠️ Mismatch` comparison table.

### 🔧 Page 3 — Phase I: Preprocessing
- One-click pipeline execution: `Run Preprocessing Pipeline`
- **Cleaned Data tab:** side-by-side raw text vs. `cleaned_text` vs. `processed_text`
- **Feature Stats tab:** box plots of engineered features (word count, exclamation count, etc.)
- **Slang Examples tab:** full table of e-commerce slang normalisations

### 📊 Page 4 — Phase II: VADER Sentiment
- KPI row: Dissatisfaction Index · % Dissatisfied · % Severely Dissatisfied · Avg Compound
- Dissatisfaction Gauge (0–100) + Sentiment distribution pie chart
- Compound vs. Rating scatter plot (model validation chart)
- Department-level dissatisfaction bar chart
- **Review Explorer:** filter reviews by severity class and inspect raw text

### 🗂️ Page 5 — Phase II: LDA Topic Modeling
- Adjustable sliders: number of topics (3–10) and training passes (5–20)
- **Topic Keywords tab:** 6 insight cards with top-8 words + probability weights per topic
- **Heatmap tab:** Topic × Star Rating matrix showing average dissatisfaction per cell
- **Topic Distribution tab:** bar chart + pie chart of topic share across all reviews
- **Labelled Reviews tab:** filterable data table showing per-review topic assignments

### 🎭 Page 6 — Phase II: RoBERTa Sarcasm Detection
- Model: `cardiffnlp/twitter-roberta-base-irony` (~500 MB, downloaded on first run)
- Adjustable sample slider (100–3,000 reviews)
- Sarcasm Donut chart + Irony Probability Histogram
- **Sarcastic Reviews table:** sorted by irony probability (highest first)
- **VADER vs RoBERTa Disagreement Table:** reviews VADER got wrong that RoBERTa corrected

### 📈 Page 7 — Phase III: Analytics Dashboard
- 5-metric Executive Summary banner
- Integrated Core Sentiment Overview (Gauge + Rating Distribution + Sentiment Pie)
- Dissatisfaction Pain-Point Heatmap (full resolution)
- Department & Age demographic analysis
- Positive / Negative Word Clouds (split by star rating)
- Sarcasm Integration Panel (conditional — appears after sarcasm page is run)
- **Export Section:** download full analysed CSV, sarcasm results CSV, and KPI summary CSV

---

## 🔬 Key Algorithms Explained

### Dissatisfaction Score Formula
```python
# src/sentiment_vader.py — Line 52
Dissatisfaction Score = max(0.0, -compound) × 100

# Where compound ∈ [-1.0, +1.0] (VADER output)
# Examples:
#   compound = -0.80  →  Score =  80/100  (Severely Dissatisfied)
#   compound = -0.45  →  Score =  45/100  (Highly Dissatisfied)
#   compound = +0.60  →  Score =   0/100  (Satisfied)
```

### Severity Classification
```
Score 70–100  →  🔴 Severely Dissatisfied
Score 45–69   →  🟠 Highly Dissatisfied
Score 20–44   →  🟡 Moderately Dissatisfied
Score  5–19   →  🟢 Mildly Dissatisfied
Score  0–4    →  ✅ Satisfied
```

### Hybrid Sarcasm Score (RoBERTa + Lexical Boost)
```
                Raw Review Text
                      │
          ┌───────────┴───────────┐
          │                       │
    RoBERTa Model          Lexical Boost
    (Context-based)        (Pattern-based, only if VADER compound > 0.3)
    irony_prob = 0.356     boost = +0.600
          │                       │
          └───────────┬───────────┘
                      │
            combined = min(1.0, 0.356 + 0.600) = 0.956
                      │
          ≥ 0.55 threshold → IS SARCASTIC ✅
                      │
          Dissatisfaction Score overridden → max(60, 95.6) = 95.6/100
```

### LDA Coherence Score (C_v) Interpretation
| C_v Score | Topic Quality |
|-----------|--------------|
| > 0.70 | Excellent — topics are highly distinct |
| 0.55–0.70 | Good — topics are meaningful and interpretable |
| 0.40–0.55 | Acceptable — minor topic overlap |
| < 0.40 | Poor — increase passes or adjust num_topics |

---

## 📊 Dataset

| Attribute | Detail |
|-----------|--------|
| **Name** | Women's Clothing E-Commerce Reviews |
| **Source** | Kaggle (publicly available) |
| **Records** | 23,486 reviews |
| **Key Columns** | Review Text, Rating (1–5★), Department Name, Age, Division Name, Recommended IND |
| **Departments** | Tops, Dresses, Bottoms, Intimate, Jackets, Trend |
| **Rating Split** | 1★ (3.6%) · 2★ (6.7%) · 3★ (12.2%) · 4★ (21.6%) · 5★ (55.9%) |
| **File Location** | `dataset/Womens Clothing E-Commerce Reviews.csv` |

> The dataset is included in this repository for convenience. It is publicly available on Kaggle under its original license.

---

## ✅ Output Validation

All outputs are mathematically traceable and validated through multiple independent mechanisms:

| Output | Validation Method |
|--------|-----------------|
| Cleaned Text | Side-by-side comparison with raw text in dashboard |
| Dissatisfaction Score | Cross-validated against star ratings via scatter plot |
| Severity Class | Threshold-based classification tied to VADER compound ranges |
| LDA Topics | C_v Coherence Score (mathematical quality metric) |
| Topic Assignment | `topic_probability` score displayed per review |
| Topic × Rating Heatmap | Triple cross-validation: VADER + LDA + human star ratings |
| Sarcasm Detection | VADER vs RoBERTa Disagreement Table for manual verification |
| Live Review Output | Expected vs. Predicted match/mismatch comparison table |

For a full 21-step validation walkthrough with exact code references, see **[VALIDATION_GUIDE.md](VALIDATION_GUIDE.md)**.

---

## 📚 Documentation Index

| Document | Purpose |
|----------|---------|
| **[VALIDATION_GUIDE.md](VALIDATION_GUIDE.md)** | How every output is calculated and validated — 21 steps with code references |
| **[VIVA_PRESENTATION_GUIDE.md](VIVA_PRESENTATION_GUIDE.md)** | Academic panel Q&A guide with model answers for all likely questions |
| **[DEMONSTRATION_VIDEO_SCRIPT.md](DEMONSTRATION_VIDEO_SCRIPT.md)** | Full 20-minute demo script for video presentation |

---

## 🔑 Key Research Findings

1. **The Sarcasm Blind Spot:** Standard sentiment tools misclassify ~3.5% of reviews due to sarcasm. The hybrid RoBERTa + lexical approach recovers these hidden negative signals.

2. **Rating vs. Text Dissatisfaction Gap:** Star ratings are often "sticky" — customers give 3–4 stars but express 80/100 dissatisfaction in the text. The NLP score is more operationally sensitive than the rating column.

3. **Dominant Pain Points:** Unsupervised topic modeling reveals that **Product Quality** and **Fit/Size** are the primary drivers of severe dissatisfaction, accounting for the majority of high-dissatisfaction scores.

---

## 🔮 Limitations & Future Work

**Current Limitations:**
- VADER and RoBERTa are English-only; multilingual support would require XLM-RoBERTa
- LDA topics were pre-labelled based on domain knowledge (subjective interpretation)
- RoBERTa has a Twitter→E-commerce domain gap, partially compensated by the lexical boost layer

**Planned Future Work:**
- Fine-tune RoBERTa on e-commerce-specific sarcasm data
- Add Aspect-Based Sentiment Analysis (ABSA) to identify specific product components (e.g., *"zipper"*, *"buttons"*) within topics
- Replace keyword-based live topic classifier with zero-shot classification
- Add temporal trend analysis (monthly dissatisfaction index tracking)
- Multilingual support via XLM-RoBERTa

---

## 🙏 Acknowledgements

- **Dataset:** Women's Clothing E-Commerce Reviews — Kaggle
- **RoBERTa Model:** [cardiffnlp/twitter-roberta-base-irony](https://huggingface.co/cardiffnlp/twitter-roberta-base-irony) — Cardiff NLP Group
- **VADER:** Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. ICWSM-2014.
- **Gensim LDA:** Řehůřek, R. & Sojka, P. (2010). Software Framework for Topic Modelling with Large Corpora. LREC 2010.

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` for details.

---

*DSA Final Year Research Project — Quantifying Customer Dissatisfaction: A Hybrid NLP Approach*
*Built with: Python · Streamlit · NLTK · Gensim · HuggingFace Transformers · Plotly*

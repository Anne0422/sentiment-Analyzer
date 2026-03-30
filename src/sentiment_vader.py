"""
Phase II – VADER Sentiment Analysis Engine
Applies a refined VADER rule-set tailored for e-commerce review nuances.
Computes per-review Dissatisfaction Scores and classifies severity.
"""
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)

# ── Custom E-Commerce Lexicon Extension ─────────────────────────────────────
CUSTOM_ECOMMERCE_LEXICON = {
    # Negative terms (e-commerce specific)
    'overpriced': -3.2, 'defective': -3.5, 'disappointed': -2.8,
    'misleading': -3.0, 'unwearable': -3.2, 'scratchy': -2.5,
    'unflattering': -2.3, 'cheaply': -2.5, 'flimsy': -2.4,
    'shrunk': -2.2, 'faded': -2.0, 'pilling': -2.1, 'itchy': -2.3,
    'returned': -1.8, 'returning': -1.9, 'refund': -2.0,
    'damaged': -3.0, 'broken': -2.8, 'terrible': -3.5,
    'horrible': -3.4, 'awful': -3.3, 'dreadful': -3.2,
    'unacceptable': -3.1, 'waste': -2.8, 'useless': -2.7,
    'complaint': -2.5, 'regret': -2.4, 'mistake': -2.0,
    'shrinks': -2.2, 'snag': -2.0, 'snagged': -2.1,
    'runs small': -1.5, 'runs large': -1.5,
    # Positive terms (e-commerce specific)
    'flattering': 2.8, 'well-made': 3.0, 'comfortable': 2.5,
    'fits perfectly': 3.0, 'true to size': 2.0, 'high quality': 3.2,
    'durable': 2.5, 'stunning': 3.2, 'gorgeous': 3.0,
    'elegant': 2.8, 'versatile': 2.3, 'lightweight': 2.0,
    'breathable': 2.2, 'silky': 2.5, 'luxurious': 3.0,
    'exactly as described': 2.5, 'exceeded expectations': 3.5,
}


def get_vader_analyzer() -> SentimentIntensityAnalyzer:
    """Build VADER SIA with custom e-commerce lexicon entries."""
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(CUSTOM_ECOMMERCE_LEXICON)
    return sia


def analyze_single(text: str, sia: SentimentIntensityAnalyzer) -> dict:
    """Return VADER scores + dissatisfaction score for one review."""
    if not isinstance(text, str) or not text.strip():
        return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0,
                'dissatisfaction_score': 0.0, 'sentiment_class': 'Neutral'}
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    # Dissatisfaction Score: 0–100 scale, peaks at compound = -1.0
    dis_score = round(max(0.0, -compound) * 100, 2)
    return {
        'compound': round(compound, 4),
        'vader_pos': round(scores['pos'], 4),
        'vader_neu': round(scores['neu'], 4),
        'vader_neg': round(scores['neg'], 4),
        'dissatisfaction_score': dis_score,
        'sentiment_class': classify_dissatisfaction(dis_score),
    }


def classify_dissatisfaction(score: float) -> str:
    """Map numeric dissatisfaction score to categorical label."""
    if score >= 70:
        return 'Severely Dissatisfied'
    elif score >= 45:
        return 'Highly Dissatisfied'
    elif score >= 20:
        return 'Moderately Dissatisfied'
    elif score >= 5:
        return 'Mildly Dissatisfied'
    else:
        return 'Satisfied'


def batch_analyze_vader(df: pd.DataFrame, text_col: str = 'Review Text') -> pd.DataFrame:
    """Apply VADER analysis to every row and return enriched DataFrame."""
    sia = get_vader_analyzer()
    records = df[text_col].apply(lambda t: analyze_single(t, sia))
    results = pd.DataFrame(records.tolist())
    return pd.concat([df.reset_index(drop=True), results], axis=1)


def compute_business_kpis(df: pd.DataFrame) -> dict:
    """Aggregate business-level dissatisfaction KPIs from analyzed DataFrame."""
    if 'dissatisfaction_score' not in df.columns:
        return {}
    neg_df = df[df['dissatisfaction_score'] > 5]
    return {
        'overall_dissatisfaction_index': round(df['dissatisfaction_score'].mean(), 2),
        'pct_dissatisfied': round((df['dissatisfaction_score'] > 5).mean() * 100, 1),
        'pct_severely_dissatisfied': round(
            (df['dissatisfaction_score'] >= 70).mean() * 100, 1),
        'avg_compound_score': round(df['compound'].mean(), 4),
        'most_negative_review': df.loc[df['dissatisfaction_score'].idxmax(),
                                       'Review Text'] if len(df) else '',
        'dissatisfaction_by_department': (
            df.groupby('Department Name')['dissatisfaction_score'].mean()
            .round(2).to_dict() if 'Department Name' in df.columns else {}
        ),
        'dissatisfaction_by_rating': (
            df.groupby('Rating')['dissatisfaction_score'].mean()
            .round(2).to_dict() if 'Rating' in df.columns else {}
        ),
    }


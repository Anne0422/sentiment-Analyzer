"""
Phase II – Transformer-Based Sarcasm / Irony Detector
Uses cardiffnlp/twitter-roberta-base-irony (RoBERTa) + lexical heuristics
to detect nuanced customer feedback that VADER misses – especially sarcasm & irony.

Fix notes
---------
* `return_all_scores=True` is deprecated in newer transformers; replaced with `top_k=None`
  so the pipeline always returns ALL class probabilities as a list-of-dicts.
* Added `_lexical_sarcasm_boost()` to handle e-commerce sarcasm patterns that the
  Twitter-trained RoBERTa model under-scores (quoted praise, explicit negative phrases,
  rhetorical 'if you enjoy wasting money' constructions, etc.).
* `detect_single` and `batch_detect_sarcasm` now return the combined hybrid score.
"""
import re
import pandas as pd
import numpy as np

# ── Model Configuration ──────────────────────────────────────────────────────
MODEL_NAME = "cardiffnlp/twitter-roberta-base-irony"
MAX_LENGTH = 128   # RoBERTa input cap (Twitter-tuned model)
BATCH_SIZE = 16

# Label mapping for the irony model
IRONY_LABELS = {0: 'non_irony', 1: 'irony'}
IRONY_THRESHOLD = 0.55   # Probability cutoff for flagging as sarcastic

# ── Lexical sarcasm signals (used by _lexical_sarcasm_boost) ─────────────────
# Explicit negative phrases that clash with a high VADER positive score
_NEG_PHRASES = [
    'poor craftsmanship', 'poor quality', 'poor service', 'poor material',
    'fell apart', 'falling apart', 'falls apart', 'came apart',
    'wasting your money', 'waste of money', 'wasting money', 'money down the drain',
    'not worth', 'not worth it', 'overpriced', 'rip off', 'ripoff',
    'worst purchase', 'worst product', 'biggest mistake',
    'do not buy', "don't buy", 'never buying', 'avoid this',
    'defective', 'broken', 'damaged', 'ruined', 'useless',
    'garbage', 'trash', 'rubbish', 'junk', 'scam', 'fraud',
    'terrible quality', 'horrible quality', 'awful quality',
    'fell off', 'buttons fell', 'zipper broke', 'seam ripped',
]


def _lexical_sarcasm_boost(text: str, compound: float) -> float:
    """
    Compute an additive boost to RoBERTa's irony_prob based on lexical patterns.

    Only activates when VADER compound > 0.3 (i.e. VADER is misleadingly positive).
    Returns a float in [0.0, 0.60].

    Pattern weights (capped at 0.60 total):
      • Quoted word(s)  e.g. 'beautiful'        → +0.22  (very strong signal)
      • Explicit neg phrases (per phrase, cap 3) → +0.12 each (max +0.36)
      • Rhetorical "if you enjoy wasting…"       → +0.25
      • "Oh, what a [positive]" opener           → +0.18
      • "truly a masterpiece/example of [neg]"   → +0.22
    """
    if compound <= 0.3:          # VADER already looks negative — no boost needed
        return 0.0

    t = text.lower()
    boost = 0.0

    # 1. Quoted praise words — e.g.  "what a 'beautiful' surprise"
    if re.search(r"['\"][a-z]{2,}['\"]", text):
        boost += 0.22

    # 2. Explicit negative phrases at odds with high VADER compound
    neg_hits = sum(1 for ph in _NEG_PHRASES if ph in t)
    boost += min(neg_hits * 0.12, 0.36)   # cap at 0.36

    # 3. Rhetorical "if you enjoy/like/love [negative activity]"
    if re.search(r'if you (enjoy|like|love|want) (wasting|losing|throwing|spending)', t):
        boost += 0.25

    # 4. Classic sarcasm opener: "Oh, what a [positive noun]" + negative content
    if re.search(r'\b(oh[,.]?\s+)?what a.{1,50}(surprise|treat|delight|joy|pleasure|dream)', t):
        if neg_hits > 0:
            boost += 0.18

    # 5. "Truly a masterpiece/example of [negative]"
    if re.search(
        r'truly (a |an )?(masterpiece|example|testament|showcase|display).{1,60}(poor|bad|terrible|horrible|awful)',
        t
    ):
        boost += 0.22

    return min(boost, 0.60)   # hard cap


def _parse_pipeline_scores(raw_output_item):
    """
    Robustly extract a score_map dict from a single pipeline output item.

    Handles two formats produced by different transformers versions:
      • top_k=None  → item is a LIST of {label, score} dicts  ← correct/new
      • return_all_scores=True (deprecated) → item may be a single DICT  ← broken
    """
    if isinstance(raw_output_item, list):
        # Correct format: [{label: 'non_irony', score: ...}, {label: 'irony', score: ...}]
        return {r['label']: r['score'] for r in raw_output_item}
    elif isinstance(raw_output_item, dict):
        # Old / broken format: single top-1 dict → compute the other class
        label = raw_output_item.get('label', 'non_irony')
        score = raw_output_item.get('score', 0.5)
        other = 'irony' if label == 'non_irony' else 'non_irony'
        return {label: score, other: round(1.0 - score, 6)}
    return {'non_irony': 1.0, 'irony': 0.0}


def _preprocess_for_roberta(text: str) -> str:
    """Minimal pre-processing for RoBERTa: preserve punctuation & case."""
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+|www\S+', '@URL', text)
    text = re.sub(r'@\w+', '@USER', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:512]


def load_roberta_pipeline():
    """
    Load the HuggingFace pipeline for irony detection.
    Returns None if transformers/torch are unavailable.
    Uses top_k=None (replaces deprecated return_all_scores=True) so ALL class
    probabilities are always returned.
    """
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        pipe = pipeline(
            task='text-classification',
            model=model,
            tokenizer=tokenizer,
            top_k=None,        # ← returns all class scores (replaces return_all_scores=True)
            truncation=True,
            max_length=MAX_LENGTH,
        )
        return pipe
    except Exception as e:
        print(f"[Sarcasm Detector] Model load failed: {e}")
        return None


def detect_single(text: str, pipe, compound: float = 0.0) -> dict:
    """
    Run hybrid sarcasm detection on a single review text.

    Combines:
      1. RoBERTa irony probability  (model-based)
      2. Lexical sarcasm boost      (pattern-based, compensates for Twitter→e-commerce gap)

    Parameters
    ----------
    text     : raw review string
    pipe     : loaded HuggingFace pipeline (or None)
    compound : VADER compound score for this text (used to gate the lexical boost)
    """
    if pipe is None:
        return {'irony_prob': 0.0, 'non_irony_prob': 1.0,
                'is_sarcastic': False, 'lexical_boost': 0.0}

    cleaned = _preprocess_for_roberta(text)
    if not cleaned:
        return {'irony_prob': 0.0, 'non_irony_prob': 1.0,
                'is_sarcastic': False, 'lexical_boost': 0.0}

    # ── Step 1: RoBERTa model score ───────────────────────────────────────────
    try:
        raw = pipe(cleaned)
        # With top_k=None: pipe(text) → [[{label, score}, ...]]
        # We want the inner list
        score_map = _parse_pipeline_scores(raw[0])
        roberta_irony = round(score_map.get('irony', 0.0), 4)
        non_irony_prob = round(score_map.get('non_irony', 1.0), 4)
    except Exception:
        roberta_irony  = 0.0
        non_irony_prob = 1.0

    # ── Step 2: Lexical boost (compensates for domain gap) ───────────────────
    boost = _lexical_sarcasm_boost(text, compound)
    combined = round(min(1.0, roberta_irony + boost), 4)

    return {
        'irony_prob':    combined,          # hybrid score used for decisions
        'roberta_irony': roberta_irony,     # raw model score (for transparency)
        'lexical_boost': round(boost, 4),   # boost applied
        'non_irony_prob': non_irony_prob,
        'is_sarcastic':  combined >= IRONY_THRESHOLD,
    }


def batch_detect_sarcasm(texts: pd.Series, pipe,
                         batch_size: int = BATCH_SIZE,
                         progress_callback=None,
                         compounds: pd.Series = None) -> pd.DataFrame:
    """
    Run hybrid sarcasm detection on a Series of texts in batches.

    Parameters
    ----------
    texts     : Series of raw review strings
    pipe      : loaded HuggingFace pipeline
    compounds : optional Series of VADER compound scores (same index as texts)
                used to gate the lexical boost; defaults to 0.0 for all rows
    progress_callback(current, total) is called after each batch.
    Returns DataFrame with irony_prob, non_irony_prob, is_sarcastic columns.
    """
    records = []
    texts_list    = texts.tolist()
    compounds_list = (compounds.tolist() if compounds is not None
                      else [0.0] * len(texts_list))
    total = len(texts_list)

    for start in range(0, total, batch_size):
        batch_texts     = texts_list[start:start + batch_size]
        batch_compounds = compounds_list[start:start + batch_size]

        cleaned_batch = [_preprocess_for_roberta(t) or 'neutral review'
                         for t in batch_texts]

        if pipe is not None:
            try:
                raw_outputs = pipe(cleaned_batch)   # list of items (one per text)
                for raw_item, orig_text, cmp in zip(raw_outputs, batch_texts, batch_compounds):
                    score_map  = _parse_pipeline_scores(raw_item)
                    rob_irony  = round(score_map.get('irony', 0.0), 4)
                    boost      = _lexical_sarcasm_boost(orig_text, cmp)
                    combined   = round(min(1.0, rob_irony + boost), 4)
                    records.append({
                        'irony_prob':     combined,
                        'roberta_irony':  rob_irony,
                        'lexical_boost':  round(boost, 4),
                        'non_irony_prob': round(score_map.get('non_irony', 1.0), 4),
                        'is_sarcastic':   combined >= IRONY_THRESHOLD,
                    })
            except Exception:
                for orig_text, cmp in zip(batch_texts, batch_compounds):
                    boost = _lexical_sarcasm_boost(orig_text, cmp)
                    combined = round(min(1.0, boost), 4)
                    records.append({'irony_prob': combined, 'roberta_irony': 0.0,
                                    'lexical_boost': round(boost, 4),
                                    'non_irony_prob': round(1.0 - combined, 4),
                                    'is_sarcastic': combined >= IRONY_THRESHOLD})
        else:
            for orig_text, cmp in zip(batch_texts, batch_compounds):
                boost = _lexical_sarcasm_boost(orig_text, cmp)
                combined = round(min(1.0, boost), 4)
                records.append({'irony_prob': combined, 'roberta_irony': 0.0,
                                'lexical_boost': round(boost, 4),
                                'non_irony_prob': round(1.0 - combined, 4),
                                'is_sarcastic': combined >= IRONY_THRESHOLD})

        if progress_callback:
            progress_callback(min(start + batch_size, total), total)

    return pd.DataFrame(records)


def get_sarcasm_kpis(df: pd.DataFrame) -> dict:
    """Compute sarcasm-related KPIs from analyzed DataFrame."""
    if 'is_sarcastic' not in df.columns:
        return {}
    sarcastic = df[df['is_sarcastic']]
    return {
        'total_analyzed': len(df),
        'sarcastic_count': int(sarcastic.shape[0]),
        'sarcasm_rate_pct': round(sarcastic.shape[0] / max(len(df), 1) * 100, 1),
        'avg_irony_prob': round(df['irony_prob'].mean(), 4),
        'high_confidence_sarcasm': int((df['irony_prob'] >= 0.75).sum()),
    }


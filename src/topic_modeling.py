"""
Phase II – LDA Topic Modeling Engine
Discovers latent dissatisfaction themes using Gensim LDA.
Topics: Fit/Size, Product Quality, Delivery, Customer Service, Value, Style/Design.
"""
import re
import pandas as pd
import numpy as np
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Topic Configuration ──────────────────────────────────────────────────────
NUM_TOPICS = 6
TOPIC_LABELS = {
    0: '📏 Fit & Size',
    1: '🏷️ Product Quality',
    2: '🚚 Delivery & Shipping',
    3: '🤝 Customer Service',
    4: '💰 Value for Money',
    5: '🎨 Style & Design',
}
TOPIC_COLORS = {
    '📏 Fit & Size': '#EF4444',
    '🏷️ Product Quality': '#F97316',
    '🚚 Delivery & Shipping': '#EAB308',
    '🤝 Customer Service': '#8B5CF6',
    '💰 Value for Money': '#06B6D4',
    '🎨 Style & Design': '#10B981',
}

LDA_STOP_WORDS = set(stopwords.words('english')) | {
    'dress', 'top', 'shirt', 'wear', 'wearing', 'wore', 'worn',
    'bought', 'ordered', 'purchase', 'item', 'product', 'one',
    'also', 'would', 'get', 'got', 'like', 'look', 'looked',
    'really', 'very', 'just', 'even', 'still', 'back', 'could',
    'color', 'colour', 'size', 'petite', 'small', 'medium', 'large',
}


def tokenize_for_lda(text: str) -> list:
    """Tokenize and filter tokens for LDA corpus."""
    if not isinstance(text, str):
        return []
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    tokens = word_tokenize(text)
    return [t for t in tokens if t not in LDA_STOP_WORDS and len(t) > 3]


def build_corpus(texts: pd.Series):
    """Build Gensim dictionary and BoW corpus from text series."""
    tokenized = [tokenize_for_lda(t) for t in texts]
    tokenized = [t for t in tokenized if len(t) >= 3]
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=5, no_above=0.6, keep_n=5000)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]
    return corpus, dictionary, tokenized


def train_lda_model(corpus, dictionary, num_topics: int = NUM_TOPICS,
                    passes: int = 10, random_state: int = 42):
    """Train Gensim LDA model with optimised hyperparameters."""
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        update_every=1,
        chunksize=200,
        passes=passes,
        alpha='auto',
        eta='auto',
        per_word_topics=True,
    )
    return lda_model


def get_coherence_score(lda_model, tokenized_texts, dictionary) -> float:
    """Compute C_v coherence score for the trained model."""
    try:
        cm = CoherenceModel(model=lda_model, texts=tokenized_texts,
                            dictionary=dictionary, coherence='c_v')
        return round(cm.get_coherence(), 4)
    except Exception:
        return 0.0


def get_topic_keywords(lda_model, num_words: int = 10) -> dict:
    """Return top keywords per topic as {topic_label: [(word, prob), ...]}."""
    result = {}
    for idx in range(lda_model.num_topics):
        label = TOPIC_LABELS.get(idx, f'Topic {idx}')
        words = lda_model.show_topic(idx, topn=num_words)
        result[label] = words
    return result


def get_dominant_topic(lda_model, bow_doc) -> tuple:
    """Return (dominant_topic_id, probability) for a single document."""
    topic_dist = lda_model.get_document_topics(bow_doc)
    if not topic_dist:
        return 0, 0.0
    dominant = max(topic_dist, key=lambda x: x[1])
    return dominant[0], round(dominant[1], 4)


def assign_topics_to_df(df: pd.DataFrame, lda_model, corpus) -> pd.DataFrame:
    """Add dominant_topic, topic_label, and topic_probability columns to df."""
    df = df.copy()
    topic_ids, topic_probs = [], []
    for bow_doc in corpus:
        tid, tprob = get_dominant_topic(lda_model, bow_doc)
        topic_ids.append(tid)
        topic_probs.append(tprob)
    # Align lengths (tokenized corpus may be shorter than df)
    pad = len(df) - len(topic_ids)
    if pad > 0:
        topic_ids += [0] * pad
        topic_probs += [0.0] * pad
    df['dominant_topic_id'] = topic_ids[:len(df)]
    df['topic_probability'] = topic_probs[:len(df)]
    df['topic_label'] = df['dominant_topic_id'].map(TOPIC_LABELS)
    return df


def get_topic_dissatisfaction_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a pivot table: avg dissatisfaction score per topic × rating."""
    if 'topic_label' not in df.columns or 'dissatisfaction_score' not in df.columns:
        return pd.DataFrame()
    pivot = df.pivot_table(
        values='dissatisfaction_score',
        index='topic_label',
        columns='Rating',
        aggfunc='mean',
    ).round(1)
    return pivot


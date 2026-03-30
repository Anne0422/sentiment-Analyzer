"""
Phase I: Intelligent Data Acquisition & Engineering
Handles cleaning, feature engineering, noise reduction, lemmatization,
and domain-specific slang for Women's E-Commerce Reviews.
"""
import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Download required NLTK resources (silent)
for resource in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                 'vader_lexicon', 'punkt_tab', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# ── E-Commerce Domain Slang Dictionary ──────────────────────────────────────
ECOMMERCE_SLANG = {
    r'\btts\b': 'true to size', r'\bngl\b': 'not going to lie',
    r'\btbh\b': 'to be honest', r'\bimo\b': 'in my opinion',
    r'\bimho\b': 'in my humble opinion', r'\bsz\b': 'size',
    r'\bbc\b': 'because', r'\bsmh\b': 'shaking my head',
    r'\brn\b': 'right now', r'\bfyi\b': 'for your information',
    r'\bfab\b': 'fabulous', r'\bluv\b': 'love',
    r'\bamazing\b': 'amazing', r'\bgorge\b': 'gorgeous',
    r'\bperfect\b': 'perfect', r'\bwud\b': 'would',
    r'\bpls\b': 'please', r'\bplz\b': 'please',
    r'\bdont\b': 'do not', r'\bdidnt\b': 'did not',
    r'\bwasnt\b': 'was not', r'\bisnt\b': 'is not',
    r'\bcant\b': 'cannot', r'\bwont\b': 'will not',
    r'\bdoesnt\b': 'does not', r'\bhasnt\b': 'has not',
    r'\bhavent\b': 'have not', r'\bharnt\b': 'have not',
}

# Negation words to preserve (important for sentiment accuracy)
NEGATION_WORDS = {'no', 'not', 'never', 'neither', 'nobody', 'nothing',
                  'nowhere', 'nor', "n't", 'cannot', 'without'}

STOP_WORDS = set(stopwords.words('english')) - NEGATION_WORDS

lemmatizer = WordNetLemmatizer()


def _get_wordnet_pos(treebank_tag: str) -> str:
    """Map Penn Treebank POS tags to WordNet POS tags for accurate lemmatization."""
    from nltk.corpus import wordnet
    tag_map = {'J': wordnet.ADJ, 'V': wordnet.VERB,
               'N': wordnet.NOUN, 'R': wordnet.ADV}
    return tag_map.get(treebank_tag[0], wordnet.NOUN)


def replace_slang(text: str) -> str:
    """Replace e-commerce slang abbreviations with full forms."""
    for pattern, replacement in ECOMMERCE_SLANG.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def clean_text(text: str) -> str:
    """Full cleaning pipeline for a single review text."""
    if not isinstance(text, str) or not text.strip():
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)                   # strip HTML
    text = re.sub(r'http\S+|www\S+', ' ', text)            # remove URLs
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)             # remove non-ASCII
    text = text.lower()
    text = replace_slang(text)
    text = re.sub(r"'s\b", '', text)                        # possessives
    text = re.sub(r"n't\b", ' not', text)                   # contractions
    text = re.sub(r"'re\b", ' are', text)
    text = re.sub(r"'ve\b", ' have', text)
    text = re.sub(r"'ll\b", ' will', text)
    text = re.sub(r"'d\b", ' would', text)
    text = re.sub(r"'m\b", ' am', text)
    text = re.sub(r'[^a-z\s]', ' ', text)                   # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def lemmatize_text(text: str) -> str:
    """Tokenize, POS-tag, remove stopwords, and lemmatize."""
    if not text:
        return ''
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    lemmatized = [
        lemmatizer.lemmatize(word, _get_wordnet_pos(tag))
        for word, tag in tagged
        if word not in STOP_WORDS and len(word) > 2
    ]
    return ' '.join(lemmatized)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply full Phase I preprocessing pipeline to the dataframe."""
    df = df.copy()
    df.rename(columns={'Unnamed: 0': 'Review ID'}, inplace=True)
    df.drop_duplicates(subset=['Review Text'], inplace=True)
    df.dropna(subset=['Review Text'], inplace=True)
    df['Review Text'] = df['Review Text'].astype(str).str.strip()
    df = df[df['Review Text'].str.len() > 10].reset_index(drop=True)

    df['cleaned_text'] = df['Review Text'].apply(clean_text)
    df['processed_text'] = df['cleaned_text'].apply(lemmatize_text)
    df = engineer_features(df)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived feature columns for analysis. Works with any CSV that has 'Review Text'."""
    df['review_length'] = df['Review Text'].str.len()
    df['word_count'] = df['Review Text'].str.split().str.len()
    # 'Title' column is optional (present in built-in dataset, may be absent in uploads)
    if 'Title' in df.columns:
        df['has_title'] = df['Title'].notna().astype(int)
    else:
        df['has_title'] = 0
    df['exclamation_count'] = df['Review Text'].str.count(r'\!')
    df['question_count'] = df['Review Text'].str.count(r'\?')
    df['uppercase_ratio'] = (
        df['Review Text'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
        / df['review_length'].replace(0, 1)
    )
    # 'Rating' column is optional — uploaded CSVs may not have star ratings
    if 'Rating' in df.columns:
        df['sentiment_label'] = df['Rating'].map({
            1: 'Very Negative', 2: 'Negative',
            3: 'Neutral', 4: 'Positive', 5: 'Very Positive'
        })
        df['is_negative'] = (df['Rating'] <= 2).astype(int)
    else:
        df['sentiment_label'] = 'Unknown'
        df['is_negative'] = 0
    return df


def get_preprocessing_stats(original: pd.DataFrame, processed: pd.DataFrame) -> dict:
    """Return a summary dict comparing raw vs processed data."""
    has_is_negative = 'is_negative' in processed.columns
    return {
        'original_rows': len(original),
        'processed_rows': len(processed),
        'rows_removed': len(original) - len(processed),
        'null_reviews_dropped': int(original['Review Text'].isna().sum()),
        'avg_word_count': round(processed['word_count'].mean(), 1),
        'avg_review_length': round(processed['review_length'].mean(), 1),
        'negative_reviews': int(processed['is_negative'].sum()) if has_is_negative else 'N/A',
        'negative_pct': round(processed['is_negative'].mean() * 100, 1) if has_is_negative else 0,
    }


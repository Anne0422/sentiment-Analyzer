"""
DSA Research Project – Quantifying Customer Dissatisfaction
Streamlit Interactive Dashboard  |  3-Phase Hybrid NLP System
Phase I  : Intelligent Data Acquisition & Engineering
Phase II : Hybrid ML/NLP Engine (VADER + LDA + RoBERTa)
Phase III: Advanced Analytics Dashboard
"""
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import os

# ── Page configuration (MUST be first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Customer Dissatisfaction Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Light theme overrides ── */
  .stApp { background-color: #F8FAFC; color: #1E293B; }
  section[data-testid="stSidebar"] {
      background-color: #EEF2FF;
      border-right: 1px solid #C7D2FE;
  }

  /* Metric cards */
  .metric-card {
      background: #FFFFFF; border-radius: 12px; padding: 20px 24px;
      border-left: 4px solid #4F46E5; margin-bottom: 12px;
      box-shadow: 0 2px 8px rgba(79,70,229,0.10);
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #4F46E5; }
  .metric-label { font-size: 0.85rem; color: #64748B; margin-top: 4px; }

  /* Phase badges */
  .phase-badge {
      display: inline-block; padding: 4px 14px; border-radius: 20px;
      font-size: 0.75rem; font-weight: 600; margin-right: 6px;
  }
  .badge-1 { background: #DBEAFE; color: #1D4ED8; }
  .badge-2 { background: #D1FAE5; color: #065F46; }
  .badge-3 { background: #FFE4E6; color: #9F1239; }

  /* Section header */
  .section-header {
      font-size: 1.5rem; font-weight: 700; color: #1E293B;
      border-bottom: 2px solid #4F46E5; padding-bottom: 8px; margin-bottom: 16px;
  }

  /* Insight / info boxes */
  .insight-box {
      background: #FFFFFF; border-radius: 10px; padding: 16px;
      border: 1px solid #E2E8F0; margin: 8px 0;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  .insight-box h4 { color: #1E293B !important; margin-bottom: 6px; }
  .insight-box p, .insight-box li { color: #475569; font-size: 0.9rem; }

  /* Metric value colour */
  div[data-testid="stMetricValue"] { color: #4F46E5 !important; }
  div[data-testid="stMetricLabel"]  { color: #64748B !important; }
  div[data-testid="stMetricDelta"]  { color: #059669 !important; }

  /* Buttons */
  .stButton > button {
      background: linear-gradient(135deg, #4F46E5, #7C3AED);
      color: #FFFFFF; border: none; border-radius: 8px;
      padding: 10px 24px; font-weight: 600;
      box-shadow: 0 2px 6px rgba(79,70,229,0.30);
  }
  .stButton > button:hover { opacity: 0.92; transform: translateY(-1px); }

  /* Headings */
  h1, h2, h3, h4 { color: #1E293B !important; }

  /* Paragraph text */
  .stMarkdown p { color: #475569; }

  /* Form labels */
  .stSelectbox label,
  .stRadio label,
  .stSlider label,
  .stTextArea label  { color: #374151 !important; font-weight: 500; }

  /* Radio options */
  .stRadio div[role="radiogroup"] label { color: #374151 !important; }

  /* Sidebar text */
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stMarkdown p { color: #374151 !important; }

  /* Tabs */
  .stTabs [data-baseweb="tab"] { color: #64748B; font-weight: 500; }
  .stTabs [aria-selected="true"] {
      color: #4F46E5 !important;
      border-bottom-color: #4F46E5 !important;
      font-weight: 700;
  }

  /* DataFrames */
  .stDataFrame { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; }

  /* Info / warning / error banners */
  .stAlert { border-radius: 8px; }

  /* Sidebar navigation title */
  section[data-testid="stSidebar"] h2 { color: #4F46E5 !important; }

  /* Horizontal rule */
  hr { border-color: #E2E8F0; }
</style>
""", unsafe_allow_html=True)

# ── Lazy imports (heavy libs loaded only when needed) ────────────────────────
@st.cache_resource(show_spinner=False)
def _import_src_modules():
    from src.preprocessing import preprocess_dataframe, get_preprocessing_stats
    from src.sentiment_vader import batch_analyze_vader, compute_business_kpis
    from src.topic_modeling import (build_corpus, train_lda_model,
                                    assign_topics_to_df, get_topic_keywords,
                                    get_topic_dissatisfaction_matrix,
                                    get_coherence_score, TOPIC_LABELS, TOPIC_COLORS)
    from src.sarcasm_detector import load_roberta_pipeline, batch_detect_sarcasm, get_sarcasm_kpis
    from src.visualizations import (
        create_dissatisfaction_gauge, create_rating_distribution,
        create_sentiment_distribution, create_dissatisfaction_histogram,
        create_topic_heatmap, create_topic_bar, create_department_sentiment_bar,
        create_scatter_compound_vs_rating, create_wordcloud_image,
        create_sarcasm_donut, create_irony_prob_histogram, create_age_sentiment_box,
    )
    return dict(
        preprocess_dataframe=preprocess_dataframe,
        get_preprocessing_stats=get_preprocessing_stats,
        batch_analyze_vader=batch_analyze_vader,
        compute_business_kpis=compute_business_kpis,
        build_corpus=build_corpus,
        train_lda_model=train_lda_model,
        assign_topics_to_df=assign_topics_to_df,
        get_topic_keywords=get_topic_keywords,
        get_topic_dissatisfaction_matrix=get_topic_dissatisfaction_matrix,
        get_coherence_score=get_coherence_score,
        TOPIC_LABELS=TOPIC_LABELS,
        TOPIC_COLORS=TOPIC_COLORS,
        load_roberta_pipeline=load_roberta_pipeline,
        batch_detect_sarcasm=batch_detect_sarcasm,
        get_sarcasm_kpis=get_sarcasm_kpis,
        create_dissatisfaction_gauge=create_dissatisfaction_gauge,
        create_rating_distribution=create_rating_distribution,
        create_sentiment_distribution=create_sentiment_distribution,
        create_dissatisfaction_histogram=create_dissatisfaction_histogram,
        create_topic_heatmap=create_topic_heatmap,
        create_topic_bar=create_topic_bar,
        create_department_sentiment_bar=create_department_sentiment_bar,
        create_scatter_compound_vs_rating=create_scatter_compound_vs_rating,
        create_wordcloud_image=create_wordcloud_image,
        create_sarcasm_donut=create_sarcasm_donut,
        create_irony_prob_histogram=create_irony_prob_histogram,
        create_age_sentiment_box=create_age_sentiment_box,
    )

M = _import_src_modules()  # module dict alias

# ── Session State Initialisation ────────────────────────────────────────────
_STATE_DEFAULTS = {
    'df_raw': None, 'df_processed': None, 'df_vader': None,
    'df_topics': None, 'df_sarcasm': None,
    'lda_model': None, 'lda_corpus': None, 'lda_dictionary': None,
    'lda_tokenized': None, 'roberta_pipe': None,
    'preprocessing_done': False, 'vader_done': False,
    'topic_done': False, 'sarcasm_done': False,
    'kpis': {}, 'topic_keywords': {},
    'live_results': None,  # stores per-review live analysis results
}
for k, v in _STATE_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

DEFAULT_DATASET = os.path.join('dataset', 'Womens Clothing E-Commerce Reviews.csv')

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Dissatisfaction\nAnalyzer")
    st.markdown("---")
    page = st.radio("**Navigate**", [
        "🏠  Home & Overview",
        "📂  Data Hub",
        "🔧  Phase I – Preprocessing",
        "📊  Phase II – Sentiment (VADER)",
        "🗂️  Phase II – Topic Modeling (LDA)",
        "🎭  Phase II – Sarcasm (RoBERTa)",
        "📈  Phase III – Analytics Dashboard",
    ], label_visibility='collapsed')
    st.markdown("---")
    st.markdown("""
**Dataset**
Women's Clothing E-Commerce
`23,486 reviews · 11 columns`

**Research Phases**
<span class='phase-badge badge-1'>Phase I</span> Data Engineering
<span class='phase-badge badge-2'>Phase II</span> Hybrid NLP
<span class='phase-badge badge-3'>Phase III</span> Dashboard
""", unsafe_allow_html=True)
    st.markdown("---")
    pipeline_status = []
    pipeline_status.append("✅ Preprocessed" if st.session_state.preprocessing_done else "⏳ Preprocessing")
    pipeline_status.append("✅ VADER Done" if st.session_state.vader_done else "⏳ VADER")
    pipeline_status.append("✅ LDA Done" if st.session_state.topic_done else "⏳ LDA")
    pipeline_status.append("✅ RoBERTa Done" if st.session_state.sarcasm_done else "⏳ RoBERTa")
    st.markdown("**Pipeline Status**")
    for s in pipeline_status:
        st.markdown(f"- {s}")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME & OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
def show_home():
    st.markdown('<h1 style="color:#6366F1;">📊 Quantifying Customer Dissatisfaction</h1>',
                unsafe_allow_html=True)
    st.markdown("### DSA Research Project — Hybrid NLP Approach")
    st.markdown("""
> *"Understanding **why** customers are unhappy — not just **that** they are unhappy."*
""")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='metric-card'>
        <div class='metric-value'>23,486</div>
        <div class='metric-label'>Customer Reviews Analysed</div></div>""",
        unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='metric-card'>
        <div class='metric-value'>6</div>
        <div class='metric-label'>Dissatisfaction Topics Discovered</div></div>""",
        unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='metric-card'>
        <div class='metric-value'>3</div>
        <div class='metric-label'>NLP Engines (VADER + LDA + RoBERTa)</div></div>""",
        unsafe_allow_html=True)

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["📌 Methodology", "🔬 Research Phases", "📁 Dataset Info"])
    with tab1:
        st.markdown("""
### The Hybrid NLP Methodology
This research integrates **three complementary NLP techniques** to quantify and
categorise customer dissatisfaction in Women's Fashion E-Commerce reviews:

| Engine | Role | Output |
|--------|------|--------|
| **VADER** (Valence Aware Dictionary) | Measure *intensity* of negativity with e-commerce-tuned lexicon | Dissatisfaction Score 0–100 |
| **LDA** (Latent Dirichlet Allocation) | Discover *latent themes* – root causes of complaints | 6 Dissatisfaction Topics |
| **RoBERTa** (Transformer) | Detect *nuanced* feedback – sarcasm & irony that VADER misses | Irony Probability 0–1 |

The combined pipeline provides stakeholders with an **Actionable Dissatisfaction Index**
that goes far beyond simple positive/negative classification.
""")
    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""<div class='insight-box'>
<span class='phase-badge badge-1'>Phase I</span>
<h4>Data Acquisition & Engineering</h4>
<ul>
<li>Multi-source input (CSV / Excel / Live Text)</li>
<li>HTML & URL stripping</li>
<li>Contraction expansion</li>
<li>E-commerce slang normalisation</li>
<li>POS-aware lemmatisation</li>
<li>Feature engineering (word count, exclamation ratio…)</li>
</ul></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class='insight-box'>
<span class='phase-badge badge-2'>Phase II</span>
<h4>Core Hybrid NLP Engine</h4>
<ul>
<li><b>VADER</b> with custom e-commerce lexicon</li>
<li>Dissatisfaction Score formula: max(0, −compound)×100</li>
<li><b>LDA</b> Topic Modelling — 6 latent themes</li>
<li>Coherence score optimisation</li>
<li><b>RoBERTa</b> irony/sarcasm detection</li>
<li>cardiffnlp/twitter-roberta-base-irony</li>
</ul></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""<div class='insight-box'>
<span class='phase-badge badge-3'>Phase III</span>
<h4>Advanced Analytics Dashboard</h4>
<ul>
<li>Dissatisfaction Gauge (0–100)</li>
<li>Topic × Rating Heatmaps</li>
<li>Department Breakdown Charts</li>
<li>Sarcasm Detection Results</li>
<li>Word Clouds (positive / negative)</li>
<li>Age-Group Dissatisfaction Analysis</li>
</ul></div>""", unsafe_allow_html=True)
    with tab3:
        st.markdown("""
### Dataset: Women's Clothing E-Commerce Reviews

| Attribute | Detail |
|-----------|--------|
| **Source** | Women's E-Commerce Clothing Reviews (Kaggle) |
| **Records** | 23,486 reviews |
| **Key Columns** | Review Text, Rating (1–5), Department, Age, Recommended IND |
| **Departments** | Tops, Dresses, Bottoms, Intimate, Jackets, Trend |
| **Rating Split** | 1★ (3.6%), 2★ (6.7%), 3★ (12.2%), 4★ (21.6%), 5★ (55.9%) |

Navigate to **📂 Data Hub** to load the dataset and begin analysis.
""")
    st.info("👉 Start by going to **📂 Data Hub** in the sidebar to load data and run the pipeline.")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: DATA HUB
# ═══════════════════════════════════════════════════════════════════════════════
def show_data_hub():
    st.markdown('<div class="section-header">📂 Data Hub — Multi-Source Input</div>',
                unsafe_allow_html=True)
    input_mode = st.radio("Select Data Source", ["📁 Use Built-in Dataset",
                                                   "⬆️ Upload CSV / Excel",
                                                   "✏️ Enter Live Review Text"],
                          horizontal=True)
    df = None
    if input_mode == "📁 Use Built-in Dataset":
        if st.button("🚀 Load Women's Clothing Dataset"):
            with st.spinner("Loading dataset..."):
                df = pd.read_csv(DEFAULT_DATASET)
                st.session_state.df_raw = df
                st.session_state.preprocessing_done = False
                st.session_state.vader_done = False
                st.session_state.topic_done = False
                st.session_state.sarcasm_done = False
            st.success(f"✅ Loaded {len(df):,} reviews!")

    elif input_mode == "⬆️ Upload CSV / Excel":
        uploaded = st.file_uploader("Upload your review file", type=['csv', 'xlsx', 'xls'])
        if uploaded:
            with st.spinner("Reading file..."):
                if uploaded.name.endswith('.csv'):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded, engine='openpyxl')
                if 'Review Text' not in df.columns:
                    st.error("❌ File must have a 'Review Text' column.")
                    df = None
                else:
                    st.session_state.df_raw = df
                    for k in ['preprocessing_done','vader_done','topic_done','sarcasm_done']:
                        st.session_state[k] = False
                    st.success(f"✅ Uploaded {len(df):,} rows!")

    else:  # ── Live Review Text Analysis ──────────────────────────────────────
        st.markdown("""<div class='insight-box'>
<h4>🔬 Live Analysis Mode</h4>
<p>Enter one or more reviews below (one per line). Optionally fill in your <b>Expected Topic</b>
and <b>Expected Sentiment</b> columns so the system can validate its predictions against your
own human judgment — great for testing accuracy!</p>
</div>""", unsafe_allow_html=True)

        col_r, col_t, col_s = st.columns([3, 2, 2])
        with col_r:
            live_text = st.text_area(
                "📝 Reviews — one per line",
                height=210,
                placeholder="e.g.\nThe package arrived two weeks late and was completely damaged.\nThe fabric smells like chemicals and falls apart after one wash.\nI love this dress, it fits perfectly and looks gorgeous!")
        with col_t:
            exp_topics_raw = st.text_area(
                "🏷️ Expected Topic (one per line, optional)",
                height=210,
                placeholder="e.g.\nDelivery & Shipping\nProduct Quality\nStyle & Design")
        with col_s:
            exp_sents_raw = st.text_area(
                "💬 Expected Sentiment (one per line, optional)",
                height=210,
                placeholder="e.g.\nNegative\nNeutral-Negative (Mixed)\nPositive")

        st.caption("💡 **Available Topics:** Delivery & Shipping · Fit & Size · Product Quality · Customer Service · Value for Money · Style & Design")
        st.caption("💡 **Available Sentiments:** Positive · Neutral-Positive (Mixed) · Neutral · Neutral-Negative (Mixed) · Negative")

        # ── RoBERTa status indicator ──────────────────────────────────────────
        roberta_ready = st.session_state.get('roberta_pipe') is not None
        st.markdown("---")
        rb_col1, rb_col2 = st.columns([3, 1])
        with rb_col1:
            if roberta_ready:
                st.success("🤖 **RoBERTa Sarcasm Detector: Loaded & Ready** — sarcasm correction will be applied automatically.")
            else:
                st.warning(
                    "🤖 **RoBERTa Sarcasm Detector: Not Loaded** — VADER-only mode. "
                    "Sarcastic reviews (e.g. *'Oh I just love how it fell apart'*) may be misclassified. "
                    "Click **Load RoBERTa** to enable sarcasm correction.")
        with rb_col2:
            if not roberta_ready:
                if st.button("⬇️ Load RoBERTa", help="Downloads ~500 MB on first run, then cached"):
                    with st.spinner("⬇️ Loading RoBERTa model (first run: 2–5 min)…"):
                        pipe = M['load_roberta_pipeline']()
                    if pipe:
                        st.session_state.roberta_pipe = pipe
                        st.success("✅ RoBERTa loaded!")
                        st.rerun()
                    else:
                        st.error("❌ Load failed. Check internet & torch install.")
            else:
                if st.button("🔄 Reload RoBERTa"):
                    with st.spinner("Reloading…"):
                        pipe = M['load_roberta_pipeline']()
                    if pipe:
                        st.session_state.roberta_pipe = pipe
                        st.success("✅ Reloaded!")
                        st.rerun()
        st.markdown("---")

        if st.button("🔍 Analyse Live Reviews", type="primary"):
            lines = [l.strip() for l in live_text.strip().split('\n') if l.strip()]
            exp_topics = ([l.strip() for l in exp_topics_raw.strip().split('\n') if l.strip()]
                          if exp_topics_raw.strip() else [])
            exp_sents  = ([l.strip() for l in exp_sents_raw.strip().split('\n') if l.strip()]
                          if exp_sents_raw.strip() else [])
            if not lines:
                st.warning("⚠️ Please enter at least one review.")
            else:
                # ── Keyword-based topic classifier (works on single reviews) ──
                LIVE_TOPIC_KW = {
                    '🚚 Delivery & Shipping': [
                        'delivery','shipping','ship','shipped','arrived','package','parcel',
                        'courier','late','delay','delayed','damaged','tracking','lost',
                        'dispatch','warehouse','customs','transit'],
                    '📏 Fit & Size': [
                        'size','fit','fits','fitting','sized','small','large','tight','loose',
                        'runs','petite','length','short','long','wide','measurements','waist',
                        'chest','hips','narrow','baggy','oversized','snug'],
                    '🏷️ Product Quality': [
                        'quality','material','fabric','thin','thick','cheap','durable',
                        'stitching','seam','faded','shrunk','pilling','itchy','smell',
                        'chemical','worn','washing','texture','thread','flimsy','scratchy',
                        'fraying','peeling','broke','defective','damaged'],
                    '🤝 Customer Service': [
                        'service','return','returned','refund','exchange','support','staff',
                        'helpful','rude','response','customer','complaint','policy',
                        'representative','contacted','chat','email','resolution','helpdesk'],
                    '💰 Value for Money': [
                        'price','expensive','cheap','worth','value','overpriced','money','cost',
                        'pay','afford','budget','discount','deal','pricey','markup','priced'],
                    '🎨 Style & Design': [
                        'style','design','pattern','beautiful','elegant','cute','pretty',
                        'gorgeous','look','fashionable','aesthetic','color','colour','print',
                        'chic','trendy','flattering','lovely','adorable'],
                }

                def _predict_topic(text: str) -> str:
                    t = text.lower()
                    scores = {topic: sum(1 for kw in kws if kw in t)
                              for topic, kws in LIVE_TOPIC_KW.items()}
                    best = max(scores, key=scores.get)
                    return best if scores[best] > 0 else '🏷️ Product Quality'

                def _predict_sentiment_label(compound: float) -> str:
                    if compound >= 0.5:    return 'Positive'
                    elif compound >= 0.1:  return 'Neutral-Positive (Mixed)'
                    elif compound >= -0.1: return 'Neutral'
                    elif compound >= -0.5: return 'Neutral-Negative (Mixed)'
                    else:                  return 'Negative'

                def _match_icon(expected: str, predicted: str) -> str:
                    if expected == '—':
                        return '—'
                    for word in expected.lower().split():
                        if len(word) > 3 and word in predicted.lower():
                            return '✅ Match'
                    return '⚠️ Mismatch'

                _roberta_pipe = st.session_state.get('roberta_pipe')
                _spinner_label = ("🔄 Running Hybrid NLP Pipeline (VADER + Keyword-LDA + RoBERTa)…"
                                  if _roberta_pipe else
                                  "🔄 Running Hybrid NLP Pipeline (VADER + Keyword-LDA)…")
                with st.spinner(_spinner_label):
                    temp_df = pd.DataFrame({
                        'Review Text': lines, 'Rating': 3,
                        'Department Name': 'Unknown', 'Age': 35,
                        'Recommended IND': 0, 'Positive Feedback Count': 0,
                        'Division Name': 'Unknown', 'Class Name': 'Unknown'})
                    vader_df = M['batch_analyze_vader'](temp_df)

                    # Import detect_single for per-review RoBERTa inference
                    from src.sarcasm_detector import detect_single, IRONY_THRESHOLD

                    live_records = []
                    for i, row in vader_df.iterrows():
                        pred_topic   = _predict_topic(str(row['Review Text']))
                        vader_compound = float(row['compound'])
                        vader_sent   = _predict_sentiment_label(vader_compound)
                        dis_score    = float(row['dissatisfaction_score'])
                        dis_class    = str(row['sentiment_class'])

                        # ── RoBERTa sarcasm detection (hybrid: model + lexical boost) ──
                        if _roberta_pipe is not None:
                            sarcasm_result = detect_single(
                                str(row['Review Text']), _roberta_pipe,
                                compound=vader_compound)
                        else:
                            sarcasm_result = {'irony_prob': 0.0, 'non_irony_prob': 1.0,
                                             'is_sarcastic': False, 'lexical_boost': 0.0}

                        irony_prob   = sarcasm_result['irony_prob']
                        is_sarcastic = sarcasm_result['is_sarcastic']

                        # ── Sentiment override when RoBERTa catches VADER's mistake ──
                        # If VADER says positive/neutral-positive but RoBERTa detects irony
                        vader_misled = vader_compound > 0.1 and is_sarcastic
                        if vader_misled:
                            # Override: sarcastic review is truly negative
                            pred_sent  = 'Negative (Sarcasm 🎭)'
                            # Dissatisfaction score: irony_prob drives it (min 60 for confirmed sarcasm)
                            dis_score  = round(max(60.0, irony_prob * 100), 1)
                            dis_class  = 'High Dissatisfaction'
                        else:
                            pred_sent  = vader_sent

                        live_records.append({
                            'review':                str(row['Review Text']),
                            'exp_topic':             exp_topics[i] if i < len(exp_topics) else '—',
                            'exp_sentiment':         exp_sents[i]  if i < len(exp_sents)  else '—',
                            'pred_topic':            pred_topic,
                            'vader_sentiment':       vader_sent,
                            'pred_sentiment':        pred_sent,
                            'dissatisfaction_class': dis_class,
                            'dissatisfaction_score': dis_score,
                            'compound':              vader_compound,
                            'vader_pos':             float(row['vader_pos']),
                            'vader_neu':             float(row['vader_neu']),
                            'vader_neg':             float(row['vader_neg']),
                            'irony_prob':            irony_prob,
                            'roberta_irony':         sarcasm_result.get('roberta_irony', irony_prob),
                            'lexical_boost':         sarcasm_result.get('lexical_boost', 0.0),
                            'is_sarcastic':          is_sarcastic,
                            'vader_misled':          vader_misled,
                            'roberta_used':          _roberta_pipe is not None,
                        })

                st.session_state.df_raw = temp_df
                st.session_state.live_results = live_records
                for k in ['preprocessing_done','vader_done','topic_done','sarcasm_done']:
                    st.session_state[k] = False
                st.success(f"✅ Analysed {len(live_records)} review(s) — see detailed results below!")
                st.rerun()

        # ── Live Results Display ──────────────────────────────────────────────
        if st.session_state.get('live_results'):
            results = st.session_state.live_results

            def _match_icon_display(expected: str, predicted: str) -> str:
                if expected == '—':
                    return '—'
                for word in expected.lower().split():
                    if len(word) > 3 and word in predicted.lower():
                        return '✅ Match'
                return '⚠️ Mismatch'

            st.markdown("---")
            st.markdown("### 🔬 Live Analysis Results")
            _any_roberta = any(r.get('roberta_used') for r in results)
            _engine_label = "VADER + Keyword-LDA + RoBERTa Sarcasm Correction" if _any_roberta else "VADER + Keyword-based Topic Classifier"
            st.markdown(f"**{len(results)} review(s) analysed** using {_engine_label}")
            if not _any_roberta:
                st.info("💡 Load RoBERTa above to enable sarcasm detection and automatic sentiment correction.")

            # ── Summary comparison table ──────────────────────────────────────
            table_rows = []
            for r in results:
                sarcasm_col = ('🎭 Sarcasm' if r.get('is_sarcastic') else
                               ('✅ OK' if r.get('roberta_used') else '—'))
                table_rows.append({
                    'Review (excerpt)':   (r['review'][:60]+'…' if len(r['review'])>60 else r['review']),
                    'Predicted Topic':    r['pred_topic'],
                    'Expected Topic':     r['exp_topic'],
                    'Topic ✓':            _match_icon_display(r['exp_topic'],    r['pred_topic']),
                    'Predicted Sentiment':r['pred_sentiment'],
                    'Expected Sentiment': r['exp_sentiment'],
                    'Sentiment ✓':        _match_icon_display(r['exp_sentiment'], r['pred_sentiment']),
                    'Sarcasm (RoBERTa)':  sarcasm_col,
                    'Score /100':         f"{r['dissatisfaction_score']:.1f}",
                    'Severity Class':     r['dissatisfaction_class'],
                })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, height=min(200 + len(results)*40, 420))

            # ── Per-review detail cards ───────────────────────────────────────
            st.markdown("#### 📋 Detailed Per-Review Breakdown")
            for i, r in enumerate(results):
                score  = r['dissatisfaction_score']
                badge_color = ('#DC2626' if score >= 70 else '#EA580C' if score >= 45
                               else '#CA8A04' if score >= 20 else '#059669')
                topic_match = _match_icon_display(r['exp_topic'],    r['pred_topic'])
                sent_match  = _match_icon_display(r['exp_sentiment'], r['pred_sentiment'])
                sarcasm_tag = " 🎭 SARCASM" if r.get('is_sarcastic') else ""

                with st.expander(
                    f"📄 Review {i+1}  |  Dissatisfaction: {score:.0f}/100  |  {r['dissatisfaction_class']}{sarcasm_tag}",
                    expanded=(len(results) <= 4)):
                    c1, c2, c3 = st.columns([3, 2, 2])

                    with c1:
                        st.markdown("**📝 Full Review Text:**")
                        st.info(r['review'])
                        # Sarcasm correction explanation
                        if r.get('vader_misled'):
                            st.error(
                                f"🎭 **Sarcasm Detected!** VADER was misled by positive surface language "
                                f"(compound = `{r['compound']:+.4f}`) and predicted **{r['vader_sentiment']}**. "
                                f"RoBERTa irony probability = **{r['irony_prob']:.0%}** → "
                                f"Sentiment corrected to **{r['pred_sentiment']}**.")
                        elif r.get('is_sarcastic'):
                            st.warning(
                                f"🎭 **Sarcasm Detected** (irony prob = {r['irony_prob']:.0%}). "
                                f"VADER compound `{r['compound']:+.4f}` already suggested negativity, "
                                f"so no override was needed.")

                    with c2:
                        exp_t_html = (f"<br><small style='color:#64748B'>▸ Your Expected: "
                                      f"<b>{r['exp_topic']}</b> &nbsp; {topic_match}</small>"
                                      if r['exp_topic'] != '—' else "")
                        exp_s_html = (f"<br><small style='color:#64748B'>▸ Your Expected: "
                                      f"<b>{r['exp_sentiment']}</b> &nbsp; {sent_match}</small>"
                                      if r['exp_sentiment'] != '—' else "")
                        sent_color = '#DC2626' if r.get('vader_misled') else '#4F46E5'
                        st.markdown(f"""
**🏷️ Predicted Topic**
<span style='font-size:1.05rem;font-weight:700;color:#4F46E5'>{r['pred_topic']}</span>
{exp_t_html}

**💬 Predicted Sentiment**
<span style='font-size:1.05rem;font-weight:700;color:{sent_color}'>{r['pred_sentiment']}</span>
{exp_s_html}

**🏷️ Severity Class**
<span style='font-size:1.05rem;font-weight:700;color:{badge_color}'>{r['dissatisfaction_class']}</span>
""", unsafe_allow_html=True)

                    with c3:
                        st.markdown("**📊 VADER Score Breakdown**")
                        st.markdown(f"""
| Dimension | Score |
|-----------|-------|
| 🟢 Positive | `{r['vader_pos']:.3f}` |
| ⚪ Neutral  | `{r['vader_neu']:.3f}` |
| 🔴 Negative | `{r['vader_neg']:.3f}` |
| ⚡ Compound | `{r['compound']:+.4f}` |
""")
                        if r.get('roberta_used'):
                            irony_bar_color = '#DC2626' if r['irony_prob'] >= 0.55 else '#059669'
                            rob_raw  = r.get('roberta_irony', r['irony_prob'])
                            lex_bst  = r.get('lexical_boost', 0.0)
                            st.markdown(f"""
**🤖 RoBERTa Hybrid Irony Score**

<div style='background:#F1F5F9;border-radius:8px;padding:8px 10px;margin-bottom:6px'>
  <div style='font-size:0.78rem;color:#64748B;margin-bottom:4px'>Combined Irony Probability</div>
  <div style='background:#E2E8F0;border-radius:4px;height:10px;overflow:hidden'>
    <div style='background:{irony_bar_color};width:{r["irony_prob"]*100:.0f}%;height:100%'></div>
  </div>
  <div style='font-size:0.9rem;font-weight:700;color:{irony_bar_color};margin-top:4px'>
    {r["irony_prob"]:.0%} {"🎭 Sarcasm Detected" if r.get("is_sarcastic") else "✅ Genuine"}
  </div>
  <div style='font-size:0.72rem;color:#64748B;margin-top:4px'>
    Model: {rob_raw:.0%} &nbsp;+&nbsp; Lexical boost: +{lex_bst:.0%}
  </div>
</div>
""", unsafe_allow_html=True)
                        st.markdown(
                            f"<div style='background:{badge_color};color:#fff;padding:12px 16px;"
                            f"border-radius:10px;text-align:center;margin-top:6px;'>"
                            f"<div style='font-size:0.8rem;opacity:0.9'>Dissatisfaction Score</div>"
                            f"<div style='font-size:2.2rem;font-weight:900;line-height:1.1'>"
                            f"{score:.0f}<span style='font-size:1rem'>/100</span></div>"
                            f"<div style='font-size:0.75rem;opacity:0.85;margin-top:2px'>"
                            f"{r['dissatisfaction_class']}</div></div>",
                            unsafe_allow_html=True)

    if st.session_state.df_raw is not None:
        df = st.session_state.df_raw
        st.markdown("---")
        st.markdown("### 📋 Raw Dataset Preview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Reviews", f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing Texts", int(df['Review Text'].isna().sum()))
        c4.metric("Avg Rating", f"{df['Rating'].mean():.2f}" if 'Rating' in df.columns else "N/A")
        st.dataframe(df.head(20), use_container_width=True, height=350)
        st.markdown("**Column Types**")
        st.dataframe(df.dtypes.reset_index().rename(columns={'index':'Column', 0:'Type'}),
                     use_container_width=True, height=200)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: PHASE I – PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def show_preprocessing():
    st.markdown('<div class="section-header">🔧 Phase I — Intelligent Data Engineering</div>',
                unsafe_allow_html=True)
    if st.session_state.df_raw is None:
        st.warning("⚠️ Please load data first in **📂 Data Hub**.")
        return

    if not st.session_state.preprocessing_done:
        st.markdown("Click **Run Preprocessing** to clean & engineer features on the raw dataset.")
        if st.button("⚙️ Run Preprocessing Pipeline"):
            with st.spinner("🔄 Cleaning text, lemmatising, engineering features…"):
                df_raw = st.session_state.df_raw
                df_proc = M['preprocess_dataframe'](df_raw)
                stats = M['get_preprocessing_stats'](df_raw, df_proc)
                st.session_state.df_processed = df_proc
                st.session_state.prep_stats = stats
                st.session_state.preprocessing_done = True
            st.success("✅ Preprocessing complete!")
            st.rerun()
        return

    df = st.session_state.df_processed
    stats = st.session_state.get('prep_stats', {})

    # KPI row
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rows After Cleaning", f"{stats.get('processed_rows',0):,}")
    c2.metric("Rows Removed", f"{stats.get('rows_removed',0):,}")
    c3.metric("Avg Word Count", stats.get('avg_word_count','—'))
    c4.metric("Negative Reviews", f"{stats.get('negative_reviews',0):,} ({stats.get('negative_pct',0)}%)")

    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🔍 Cleaned Data", "📊 Feature Stats", "🔤 Slang Examples"])
    with tab1:
        cols_show = [c for c in ['Review Text','cleaned_text','processed_text','Rating',
                                  'sentiment_label','word_count','review_length']
                     if c in df.columns]
        st.dataframe(df[cols_show].head(30), use_container_width=True, height=400)
    with tab2:
        import plotly.express as px
        feat_cols = ['review_length','word_count','exclamation_count',
                     'question_count','uppercase_ratio']
        feat_cols = [c for c in feat_cols if c in df.columns]
        if feat_cols:
            fig = px.box(df[feat_cols], color_discrete_sequence=['#4F46E5'])
            fig.update_layout(paper_bgcolor='#FFFFFF', plot_bgcolor='#F8FAFC',
                              font={'color':'#1E293B'}, height=360,
                              yaxis={'gridcolor':'#E2E8F0'})
            st.plotly_chart(fig, use_container_width=True, key="ph1_feat_box")
        if 'Rating' in df.columns:
            import plotly.graph_objects as go
            rat_cnt = df['Rating'].value_counts().sort_index()
            colors = ['#DC2626','#EA580C','#CA8A04','#65A30D','#059669']
            fig2 = go.Figure(go.Bar(x=[f'⭐{i}' for i in rat_cnt.index],
                                    y=rat_cnt.values, marker_color=colors,
                                    text=rat_cnt.values, textposition='outside',
                                    textfont={'color':'#1E293B'}))
            fig2.update_layout(title='Rating Distribution', paper_bgcolor='#FFFFFF',
                                plot_bgcolor='#F8FAFC', font={'color':'#1E293B'},
                                yaxis={'gridcolor':'#E2E8F0'}, height=300)
            st.plotly_chart(fig2, use_container_width=True, key="ph1_rating_bar")
    with tab3:
        st.markdown("""
**E-Commerce Slang Normalisation Examples**

| Raw Slang | Normalised Form |
|-----------|----------------|
| `tts` | true to size |
| `bc` | because |
| `imo` | in my opinion |
| `tbh` | to be honest |
| `dont` | do not |
| `cant` | cannot |
| `wont` | will not |
| `luv` | love |
| `gorge` | gorgeous |
| `fab` | fabulous |

**Preprocessing Pipeline Steps:**
1. Drop duplicates & null reviews
2. Strip HTML tags & URLs
3. Lowercase & remove non-ASCII
4. Expand contractions (`n't` → `not`, `'ve` → `have` …)
5. Replace domain slang
6. Remove special characters
7. Tokenise → POS-tag → Lemmatise
8. Remove stopwords (preserving negation words)
9. Feature engineering (length, exclamation count, uppercase ratio…)
""")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: PHASE II – VADER SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════════
def show_sentiment():
    st.markdown('<div class="section-header">📊 Phase II — VADER Sentiment Analysis</div>',
                unsafe_allow_html=True)
    if st.session_state.df_processed is None:
        st.warning("⚠️ Run **Phase I Preprocessing** first.")
        return

    if not st.session_state.vader_done:
        st.markdown("""
**VADER** (Valence Aware Dictionary and sEntiment Reasoner) is enhanced here with a
custom **e-commerce lexicon** (+/− weights for 40+ domain-specific terms like
`overpriced`, `flattering`, `defective`, `well-made`).

**Dissatisfaction Score formula:**
```
Dissatisfaction Score = max(0, −compound) × 100
```
where `compound ∈ [−1, +1]` from VADER.
""")
        if st.button("▶️ Run VADER Sentiment Analysis"):
            with st.spinner("🔄 Analysing sentiment for all reviews…"):
                df = M['batch_analyze_vader'](st.session_state.df_processed, 'Review Text')
                kpis = M['compute_business_kpis'](df)
                st.session_state.df_vader = df
                st.session_state.kpis = kpis
                st.session_state.vader_done = True
            st.success("✅ VADER Analysis complete!")
            st.rerun()
        return

    df = st.session_state.df_vader
    kpis = st.session_state.kpis

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Dissatisfaction Index", f"{kpis.get('overall_dissatisfaction_index',0):.1f}/100")
    c2.metric("% Dissatisfied", f"{kpis.get('pct_dissatisfied',0)}%")
    c3.metric("Severely Dissatisfied", f"{kpis.get('pct_severely_dissatisfied',0)}%")
    c4.metric("Avg VADER Compound", f"{kpis.get('avg_compound_score',0):.3f}")

    # ── Gauge + Distribution ─────────────────────────────────────────────────
    g1,g2 = st.columns(2)
    with g1:
        gauge = M['create_dissatisfaction_gauge'](
            kpis.get('overall_dissatisfaction_index', 0))
        st.plotly_chart(gauge, use_container_width=True, key="ph2v_gauge")
    with g2:
        pie = M['create_sentiment_distribution'](df)
        st.plotly_chart(pie, use_container_width=True, key="ph2v_sentiment_pie")

    st.markdown("---")
    col1,col2 = st.columns(2)
    with col1:
        hist = M['create_dissatisfaction_histogram'](df)
        st.plotly_chart(hist, use_container_width=True, key="ph2v_dis_hist")
    with col2:
        scatter = M['create_scatter_compound_vs_rating'](df)
        st.plotly_chart(scatter, use_container_width=True, key="ph2v_scatter")

    st.markdown("---")
    dept_bar = M['create_department_sentiment_bar'](df)
    st.plotly_chart(dept_bar, use_container_width=True, key="ph2v_dept_bar")

    # ── Department breakdown table ────────────────────────────────────────────
    if 'dissatisfaction_by_department' in kpis:
        st.markdown("### 🏢 Department-Level Dissatisfaction")
        dept_df = pd.DataFrame.from_dict(kpis['dissatisfaction_by_department'],
                                         orient='index', columns=['Avg Dissatisfaction Score'])
        dept_df = dept_df.sort_values('Avg Dissatisfaction Score', ascending=False)
        st.dataframe(dept_df.style.background_gradient(cmap='RdYlGn_r'),
                     use_container_width=True)

    # ── Review explorer ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Review Explorer")
    severity_filter = st.selectbox("Filter by dissatisfaction level",
                                    ['All', 'Severely Dissatisfied', 'Highly Dissatisfied',
                                     'Moderately Dissatisfied', 'Mildly Dissatisfied', 'Satisfied'])
    disp = df if severity_filter == 'All' else df[df['sentiment_class'] == severity_filter]
    show_cols = [c for c in ['Review Text','Rating','compound','dissatisfaction_score',
                              'sentiment_class','Department Name'] if c in disp.columns]
    st.dataframe(disp[show_cols].sort_values('dissatisfaction_score', ascending=False).head(50),
                 use_container_width=True, height=350)

    # ── Most negative review ──────────────────────────────────────────────────
    if kpis.get('most_negative_review'):
        st.markdown("### 🔴 Most Dissatisfied Review")
        st.error(kpis['most_negative_review'][:600])


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: PHASE II – TOPIC MODELING
# ═══════════════════════════════════════════════════════════════════════════════
def show_topic_modeling():
    st.markdown('<div class="section-header">🗂️ Phase II — LDA Topic Modeling</div>',
                unsafe_allow_html=True)
    if st.session_state.df_vader is None:
        st.warning("⚠️ Run **VADER Sentiment Analysis** first.")
        return

    if not st.session_state.topic_done:
        st.markdown("""
**Latent Dirichlet Allocation (LDA)** uncovers hidden dissatisfaction themes by treating
each review as a mixture of topics and each topic as a distribution of words.

**Hyperparameters:**
- Topics: **6** (Fit/Size, Product Quality, Delivery, Customer Service, Value, Style)
- Passes: **10** (full corpus iterations)
- α: `auto` (asymmetric Dirichlet prior)
- η: `auto` (word distribution prior)
""")
        num_topics = st.slider("Number of topics", min_value=3, max_value=10, value=6)
        passes = st.slider("Training passes (higher = better coherence, slower)", 5, 20, 10)
        if st.button("▶️ Run LDA Topic Modeling"):
            df = st.session_state.df_vader
            text_col = 'processed_text' if 'processed_text' in df.columns else 'Review Text'
            with st.spinner("🔄 Building corpus & training LDA model…"):
                corpus, dictionary, tokenized = M['build_corpus'](df[text_col])
                lda_model = M['train_lda_model'](corpus, dictionary, num_topics, passes)
                coherence = M['get_coherence_score'](lda_model, tokenized, dictionary)
                topic_kws = M['get_topic_keywords'](lda_model, num_words=12)
                df_topics = M['assign_topics_to_df'](df, lda_model, corpus)
                st.session_state.lda_model = lda_model
                st.session_state.lda_corpus = corpus
                st.session_state.lda_dictionary = dictionary
                st.session_state.lda_tokenized = tokenized
                st.session_state.df_topics = df_topics
                st.session_state.topic_keywords = topic_kws
                st.session_state.coherence_score = coherence
                st.session_state.topic_done = True
            st.success(f"✅ LDA complete! Coherence Score (C_v): {coherence:.4f}")
            st.rerun()
        return

    df = st.session_state.df_topics
    topic_kws = st.session_state.topic_keywords
    coherence = st.session_state.get('coherence_score', 0.0)

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Topics Discovered", len(topic_kws))
    c2.metric("Coherence Score (C_v)", f"{coherence:.4f}")
    c3.metric("Reviews Assigned", f"{df['topic_label'].notna().sum():,}")
    c4.metric("Dominant Topic", df['topic_label'].mode()[0] if 'topic_label' in df.columns else "—")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🔑 Topic Keywords", "🔥 Heatmap",
                                       "📊 Topic Distribution", "📋 Labelled Reviews"])
    with tab1:
        cols = st.columns(3)
        for i, (label, words) in enumerate(topic_kws.items()):
            with cols[i % 3]:
                word_str = ' · '.join([f"**{w}** ({p:.3f})" for w, p in words[:8]])
                st.markdown(f"""<div class='insight-box'>
<h4>{label}</h4>
<p style='font-size:0.85rem;color:#475569'>{word_str}</p>
</div>""", unsafe_allow_html=True)

    with tab2:
        from src.topic_modeling import get_topic_dissatisfaction_matrix
        pivot = get_topic_dissatisfaction_matrix(df)
        if not pivot.empty:
            hmap = M['create_topic_heatmap'](pivot)
            st.plotly_chart(hmap, use_container_width=True, key="ph2t_hmap")
            st.markdown("""
> **How to read:** Each cell shows the **average dissatisfaction score** for reviews
> assigned to that topic at that star rating. Darker red = more dissatisfied.
""")
        else:
            st.info("Heatmap not available — run VADER first.")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            topic_bar = M['create_topic_bar'](df)
            st.plotly_chart(topic_bar, use_container_width=True, key="ph2t_topic_bar")
        with col2:
            import plotly.express as px
            if 'topic_label' in df.columns:
                tc = df['topic_label'].value_counts()
                fig = px.pie(names=tc.index, values=tc.values, hole=0.5,
                             title='Topic Share of All Reviews')
                fig.update_layout(paper_bgcolor='#FFFFFF', height=320,
                                  font={'color':'#1E293B'})
                st.plotly_chart(fig, use_container_width=True, key="ph2t_topic_pie")

    with tab4:
        topic_filter = st.selectbox("Filter by topic", ['All'] + list(topic_kws.keys()))
        disp = df if topic_filter == 'All' else df[df['topic_label'] == topic_filter]
        show_cols = [c for c in ['Review Text','Rating','topic_label','topic_probability',
                                  'dissatisfaction_score','Department Name'] if c in disp.columns]
        st.dataframe(disp[show_cols].sort_values('dissatisfaction_score',
                                                  ascending=False).head(40),
                     use_container_width=True, height=380)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: PHASE II – SARCASM DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
def show_sarcasm():
    st.markdown('<div class="section-header">🎭 Phase II — RoBERTa Sarcasm / Irony Detection</div>',
                unsafe_allow_html=True)
    if st.session_state.df_processed is None:
        st.warning("⚠️ Run **Phase I Preprocessing** first.")
        return

    st.markdown("""
**Model:** `cardiffnlp/twitter-roberta-base-irony`
A RoBERTa transformer fine-tuned for irony detection. It catches nuanced sarcastic
reviews that rule-based VADER completely misses — e.g.:

> *"Oh great, the zipper broke after ONE wash. Absolutely love it."*

VADER scores this **positive** (love, great). RoBERTa correctly flags it as **ironic/sarcastic**.

**Threshold:** Irony probability ≥ 0.55 → flagged as sarcastic.
""")
    df_src = st.session_state.df_vader if st.session_state.df_vader is not None \
        else st.session_state.df_processed
    total = len(df_src)
    sample_n = st.slider("Number of reviews to analyse (more = slower)",
                          min_value=100, max_value=min(total, 3000),
                          value=min(500, total), step=100)

    if not st.session_state.sarcasm_done:
        st.info(f"ℹ️ The model will be downloaded (~500 MB) on first run and cached.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🤖 Load RoBERTa & Run Detection"):
                # Step 1: load model
                with st.spinner("⬇️ Loading RoBERTa model (first run may take 2–5 min)…"):
                    pipe = M['load_roberta_pipeline']()
                    st.session_state.roberta_pipe = pipe
                if pipe is None:
                    st.error("❌ Could not load RoBERTa. Check internet connection & torch install.")
                    return
                # Step 2: run detection
                sample_df = df_src.sample(min(sample_n, total), random_state=42).copy()
                progress_bar = st.progress(0, text="Detecting sarcasm…")
                def update_progress(current, total_val):
                    progress_bar.progress(current / total_val,
                                          text=f"Processed {current}/{total_val} reviews…")
                with st.spinner("🔄 Running batch inference…"):
                    compounds = (sample_df['compound'] if 'compound' in sample_df.columns
                                 else None)
                    sarcasm_df = M['batch_detect_sarcasm'](
                        sample_df['Review Text'], pipe,
                        compounds=compounds,
                        progress_callback=update_progress)
                    sample_df = sample_df.reset_index(drop=True)
                    if sarcasm_df is None or sarcasm_df.empty:
                        st.error("❌ Sarcasm detection returned no results. Please retry.")
                        return
                    sarcasm_df = sarcasm_df.reset_index(drop=True)
                    df_sarcasm = pd.concat([sample_df, sarcasm_df], axis=1)
                    kpis = M['get_sarcasm_kpis'](df_sarcasm)
                    st.session_state.df_sarcasm = df_sarcasm
                    st.session_state.sarcasm_kpis = kpis
                    st.session_state.sarcasm_done = True
                progress_bar.progress(1.0, text="Done!")
                st.success("✅ Sarcasm detection complete!")
                st.rerun()
        with col2:
            st.markdown("""<div class='insight-box'>
<h4>⚡ Why RoBERTa?</h4>
<p>Traditional lexicon methods like VADER fail to understand context.
A customer writing <i>"Oh sure, $120 for fabric that falls apart in a week — great investment!"</i>
has a positive surface tone but a deeply negative intent.
RoBERTa's bidirectional attention mechanism captures this contextual irony.</p>
</div>""", unsafe_allow_html=True)
        return

    df = st.session_state.df_sarcasm
    kpis = st.session_state.get('sarcasm_kpis', {})

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Reviews Analysed", f"{kpis.get('total_analyzed',0):,}")
    c2.metric("Sarcastic Reviews", f"{kpis.get('sarcastic_count',0):,}")
    c3.metric("Sarcasm Rate", f"{kpis.get('sarcasm_rate_pct',0)}%")
    c4.metric("High-Confidence Sarcasm", f"{kpis.get('high_confidence_sarcasm',0):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        donut = M['create_sarcasm_donut'](df)
        st.plotly_chart(donut, use_container_width=True, key="ph2s_donut")
    with col2:
        hist = M['create_irony_prob_histogram'](df)
        st.plotly_chart(hist, use_container_width=True, key="ph2s_irony_hist")

    st.markdown("---")
    st.markdown("### 🔍 Sarcastic Reviews Detected")
    if 'is_sarcastic' not in df.columns:
        st.warning("⚠️ Sarcasm column not found in results. Please re-run detection.")
    else:
        sarcastic = df[df['is_sarcastic']].sort_values('irony_prob', ascending=False)
        show_cols = [c for c in ['Review Text','Rating','irony_prob','compound',
                                  'dissatisfaction_score','Department Name'] if c in sarcastic.columns]
        st.dataframe(sarcastic[show_cols].head(30), use_container_width=True, height=380)

    st.markdown("---")
    st.markdown("### 💡 VADER vs RoBERTa Disagreement Analysis")
    st.markdown("Reviews where VADER says *satisfied* but RoBERTa detects *sarcasm:*")
    if 'compound' in df.columns and 'is_sarcastic' in df.columns:
        disagreed = df[(df['compound'] > 0.1) & (df['is_sarcastic'])]
        show_cols2 = [c for c in ['Review Text','Rating','compound','irony_prob']
                      if c in disagreed.columns]
        st.dataframe(disagreed[show_cols2].sort_values('irony_prob', ascending=False).head(20),
                     use_container_width=True, height=300)
        st.info(f"🔍 Found **{len(disagreed)}** reviews where VADER missed the sarcasm!")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: PHASE III – ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def show_dashboard():
    st.markdown('<div class="section-header">📈 Phase III — Advanced Analytics Dashboard</div>',
                unsafe_allow_html=True)

    # Determine best available dataframe
    if st.session_state.df_topics is not None:
        df = st.session_state.df_topics
    elif st.session_state.df_vader is not None:
        df = st.session_state.df_vader
    elif st.session_state.df_processed is not None:
        df = st.session_state.df_processed
    else:
        st.warning("⚠️ Load and process data first using the sidebar pages.")
        return

    kpis = st.session_state.kpis

    # ── Executive Summary Banner ──────────────────────────────────────────────
    st.markdown("## 🏆 Executive Summary")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Reviews", f"{len(df):,}")
    c2.metric("Dissatisfaction Index",
              f"{kpis.get('overall_dissatisfaction_index',0):.1f}/100"
              if kpis else "Run VADER first")
    c3.metric("% Dissatisfied",
              f"{kpis.get('pct_dissatisfied',0)}%" if kpis else "—")
    c4.metric("Avg VADER Score",
              f"{kpis.get('avg_compound_score',0):.3f}" if kpis else "—")
    c5.metric("Topics Found",
              f"{len(st.session_state.topic_keywords)}" if st.session_state.topic_done else "—")

    st.markdown("---")

    # ── Row 1: Gauge + Rating Distribution + Sentiment Donut ─────────────────
    if st.session_state.vader_done:
        st.markdown("### 📊 Core Sentiment Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            gauge = M['create_dissatisfaction_gauge'](
                kpis.get('overall_dissatisfaction_index', 0),
                "Overall Dissatisfaction Index")
            st.plotly_chart(gauge, use_container_width=True, key="dash_gauge")
        with col2:
            rating_fig = M['create_rating_distribution'](df)
            st.plotly_chart(rating_fig, use_container_width=True, key="dash_rating_dist")
        with col3:
            pie_fig = M['create_sentiment_distribution'](df)
            st.plotly_chart(pie_fig, use_container_width=True, key="dash_sentiment_pie")

    st.markdown("---")

    # ── Row 2: Topic Heatmap (if available) ──────────────────────────────────
    if st.session_state.topic_done:
        st.markdown("### 🔥 Dissatisfaction Pain-Point Heatmap")
        from src.topic_modeling import get_topic_dissatisfaction_matrix
        pivot = get_topic_dissatisfaction_matrix(df)
        if not pivot.empty:
            hmap = M['create_topic_heatmap'](pivot)
            st.plotly_chart(hmap, use_container_width=True, key="dash_hmap")
            st.markdown("""
> **Business Insight:** Red cells represent the highest-dissatisfaction combinations.
> Action these topic × rating intersections first to reduce customer churn.
""")

    st.markdown("---")

    # ── Row 3: Department + Age analysis ─────────────────────────────────────
    st.markdown("### 🏢 Department & Demographic Analysis")
    col1, col2 = st.columns(2)
    with col1:
        dept_fig = M['create_department_sentiment_bar'](df)
        st.plotly_chart(dept_fig, use_container_width=True, key="dash_dept_bar")
    with col2:
        age_fig = M['create_age_sentiment_box'](df)
        st.plotly_chart(age_fig, use_container_width=True, key="dash_age_box")

    st.markdown("---")

    # ── Row 4: Word Clouds ────────────────────────────────────────────────────
    st.markdown("### 🔤 Word Clouds")
    wc_col1, wc_col2 = st.columns(2)
    text_col = ('processed_text' if 'processed_text' in df.columns
                else 'cleaned_text' if 'cleaned_text' in df.columns else 'Review Text')
    with wc_col1:
        neg_texts = df[df['Rating'] <= 2][text_col].dropna() if 'Rating' in df.columns \
            else df[text_col].dropna()
        if len(neg_texts) > 0:
            try:
                img_bytes = M['create_wordcloud_image'](
                    neg_texts, title='🔴 Negative Review Keywords (1–2 ★)')
                st.image(img_bytes, use_container_width=True)
            except Exception as e:
                st.info(f"Word cloud unavailable: {e}")
        else:
            st.info("No negative reviews to visualize.")
    with wc_col2:
        pos_texts = df[df['Rating'] >= 4][text_col].dropna() if 'Rating' in df.columns \
            else df[text_col].dropna()
        if len(pos_texts) > 0:
            try:
                img_bytes = M['create_wordcloud_image'](
                    pos_texts, title='🟢 Positive Review Keywords (4–5 ★)')
                st.image(img_bytes, use_container_width=True)
            except Exception as e:
                st.info(f"Word cloud unavailable: {e}")
        else:
            st.info("No positive reviews to visualize.")

    st.markdown("---")

    # ── Row 5: Scatter + Topic bar ────────────────────────────────────────────
    if st.session_state.vader_done:
        st.markdown("### 🎯 Detailed Dissatisfaction Analysis")
        col1, col2 = st.columns(2)
        with col1:
            scatter = M['create_scatter_compound_vs_rating'](df)
            st.plotly_chart(scatter, use_container_width=True, key="dash_scatter")
        with col2:
            if st.session_state.topic_done:
                topic_bar = M['create_topic_bar'](df)
                st.plotly_chart(topic_bar, use_container_width=True, key="dash_topic_bar")
            else:
                hist = M['create_dissatisfaction_histogram'](df)
                st.plotly_chart(hist, use_container_width=True, key="dash_dis_hist")

    st.markdown("---")

    # ── Row 6: Sarcasm overlay (if done) ─────────────────────────────────────
    if st.session_state.sarcasm_done and st.session_state.df_sarcasm is not None:
        st.markdown("### 🎭 Sarcasm Detection Integration")
        sc_df = st.session_state.df_sarcasm
        sarcasm_kpis = st.session_state.get('sarcasm_kpis', {})
        col1, col2 = st.columns(2)
        with col1:
            donut = M['create_sarcasm_donut'](sc_df)
            st.plotly_chart(donut, use_container_width=True, key="dash_sarcasm_donut")
        with col2:
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Sarcasm Rate", f"{sarcasm_kpis.get('sarcasm_rate_pct',0)}%")
            sc2.metric("Ironic Reviews", f"{sarcasm_kpis.get('sarcastic_count',0):,}")
            sc3.metric("VADER Missed", "See Sarcasm Page")
            st.markdown("""<div class='insight-box'>
<h4>💡 Research Finding</h4>
<p>Sarcastic reviews often carry <b>hidden dissatisfaction</b> that VADER classifies as
positive. RoBERTa's contextual understanding corrects this, providing a more
accurate overall Dissatisfaction Index when combined with VADER scores.</p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Download section ──────────────────────────────────────────────────────
    st.markdown("### 📥 Export Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Analysed Data (CSV)", csv_data,
                           file_name="dissatisfaction_analysis.csv",
                           mime='text/csv', use_container_width=True)
    with col2:
        if st.session_state.df_sarcasm is not None:
            sc_csv = st.session_state.df_sarcasm.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Sarcasm Results (CSV)", sc_csv,
                               file_name="sarcasm_detection_results.csv",
                               mime='text/csv', use_container_width=True)
    with col3:
        summary = pd.DataFrame([{
            'Metric': k,
            'Value': str(v)
        } for k, v in kpis.items() if not isinstance(v, dict)])
        st.download_button("⬇️ Download KPI Summary (CSV)",
                           summary.to_csv(index=False).encode('utf-8'),
                           file_name="kpi_summary.csv",
                           mime='text/csv', use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTER
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home & Overview":
    show_home()
elif page == "📂  Data Hub":
    show_data_hub()
elif page == "🔧  Phase I – Preprocessing":
    show_preprocessing()
elif page == "📊  Phase II – Sentiment (VADER)":
    show_sentiment()
elif page == "🗂️  Phase II – Topic Modeling (LDA)":
    show_topic_modeling()
elif page == "🎭  Phase II – Sarcasm (RoBERTa)":
    show_sarcasm()
elif page == "📈  Phase III – Analytics Dashboard":
    show_dashboard()



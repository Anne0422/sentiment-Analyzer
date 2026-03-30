"""
Phase III – Visualizations Module
All Plotly/Matplotlib charts used by the Streamlit dashboard:
Gauges, heatmaps, word clouds, distributions, topic charts, time-series.
"""
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wordcloud import WordCloud

# ── Light-Theme Colour Palette ────────────────────────────────────────────────
PALETTE = {
    'primary': '#4F46E5', 'danger': '#DC2626', 'warning': '#D97706',
    'success': '#059669', 'info': '#0284C7', 'purple': '#7C3AED',
    'bg': '#F8FAFC', 'card': '#FFFFFF', 'text': '#1E293B',
    'grid': '#E2E8F0', 'subtext': '#64748B',
}
SENTIMENT_COLORS = {
    'Severely Dissatisfied': '#DC2626', 'Highly Dissatisfied': '#EA580C',
    'Moderately Dissatisfied': '#CA8A04', 'Mildly Dissatisfied': '#65A30D',
    'Satisfied': '#059669',
}


def create_dissatisfaction_gauge(score: float, title: str = "Dissatisfaction Index") -> go.Figure:
    """Plotly speedometer gauge for dissatisfaction (0–100)."""
    color = ('#059669' if score < 20 else '#CA8A04' if score < 45
             else '#EA580C' if score < 70 else '#DC2626')
    fig = go.Figure(go.Indicator(
        mode='gauge+number+delta',
        value=score,
        title={'text': title, 'font': {'size': 18, 'color': PALETTE['text']}},
        delta={'reference': 30, 'decreasing': {'color': '#059669'},
               'increasing': {'color': '#DC2626'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': PALETTE['subtext'],
                     'tickfont': {'color': PALETTE['subtext']}},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': '#F1F5F9',
            'bordercolor': '#C7D2FE',
            'steps': [
                {'range': [0, 20],  'color': '#DCFCE7'},
                {'range': [20, 45], 'color': '#FEF9C3'},
                {'range': [45, 70], 'color': '#FFEDD5'},
                {'range': [70, 100],'color': '#FEE2E2'},
            ],
            'threshold': {'line': {'color': '#1E293B', 'width': 3},
                          'thickness': 0.8, 'value': score},
        },
        number={'suffix': '/100', 'font': {'size': 28, 'color': color}},
    ))
    fig.update_layout(paper_bgcolor=PALETTE['card'], height=280,
                      margin=dict(t=50, b=10, l=20, r=20),
                      font={'color': PALETTE['text']})
    return fig


def create_rating_distribution(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of review counts per star rating."""
    counts = df['Rating'].value_counts().sort_index()
    colors = ['#DC2626', '#EA580C', '#CA8A04', '#65A30D', '#059669']
    fig = go.Figure(go.Bar(
        y=[f'⭐ {i}' for i in counts.index],
        x=counts.values,
        orientation='h',
        marker_color=colors[:len(counts)],
        text=counts.values, textposition='outside',
        textfont={'color': PALETTE['text']},
    ))
    fig.update_layout(title='Rating Distribution', xaxis_title='Number of Reviews',
                      paper_bgcolor=PALETTE['card'], plot_bgcolor=PALETTE['bg'],
                      font={'color': PALETTE['text']}, height=280,
                      xaxis={'gridcolor': PALETTE['grid']},
                      margin=dict(t=50, b=30, l=20, r=60))
    return fig


def create_sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    """Donut chart of sentiment class proportions."""
    if 'sentiment_class' not in df.columns:
        return go.Figure()
    counts = df['sentiment_class'].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        hole=0.55, marker_colors=[SENTIMENT_COLORS.get(l, '#94A3B8') for l in counts.index],
        textinfo='label+percent', textfont_size=12,
        textfont_color=PALETTE['text'],
    ))
    fig.update_layout(title='Dissatisfaction Severity Distribution',
                      paper_bgcolor=PALETTE['card'], height=320,
                      font={'color': PALETTE['text']},
                      margin=dict(t=50, b=10, l=10, r=10),
                      showlegend=False)
    return fig


def create_dissatisfaction_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of per-review dissatisfaction scores."""
    fig = px.histogram(df, x='dissatisfaction_score', nbins=40,
                       color_discrete_sequence=[PALETTE['danger']],
                       labels={'dissatisfaction_score': 'Dissatisfaction Score'})
    fig.update_layout(title='Distribution of Dissatisfaction Scores',
                      paper_bgcolor=PALETTE['card'], plot_bgcolor=PALETTE['bg'],
                      font={'color': PALETTE['text']}, height=300,
                      xaxis={'gridcolor': PALETTE['grid']},
                      yaxis={'gridcolor': PALETTE['grid']},
                      margin=dict(t=50, b=30, l=40, r=20))
    return fig


def create_topic_heatmap(pivot_df: pd.DataFrame) -> go.Figure:
    """Heatmap of avg dissatisfaction by topic × star rating."""
    if pivot_df.empty:
        return go.Figure()
    fig = go.Figure(go.Heatmap(
        z=pivot_df.values,
        x=[f'⭐{c}' for c in pivot_df.columns],
        y=pivot_df.index.tolist(),
        colorscale='RdYlGn_r',
        text=pivot_df.values, texttemplate='%{text:.1f}',
        textfont={'color': '#1E293B'},
        hovertemplate='Topic: %{y}<br>Rating: %{x}<br>Avg Dissatisfaction: %{z:.1f}<extra></extra>',
    ))
    fig.update_layout(title='Dissatisfaction Heatmap: Topic × Rating',
                      paper_bgcolor=PALETTE['card'], plot_bgcolor=PALETTE['bg'],
                      font={'color': PALETTE['text']}, height=360,
                      margin=dict(t=60, b=40, l=140, r=20))
    return fig


def create_topic_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart of average dissatisfaction score per topic."""
    if 'topic_label' not in df.columns or 'dissatisfaction_score' not in df.columns:
        return go.Figure()
    avg = df.groupby('topic_label')['dissatisfaction_score'].mean().sort_values(ascending=True)
    fig = go.Figure(go.Bar(y=avg.index, x=avg.values, orientation='h',
                           marker_color=PALETTE['danger'],
                           text=avg.values.round(1), textposition='outside',
                           textfont={'color': PALETTE['text']}))
    fig.update_layout(title='Avg Dissatisfaction Score by Topic',
                      paper_bgcolor=PALETTE['card'], plot_bgcolor=PALETTE['bg'],
                      font={'color': PALETTE['text']}, height=320,
                      xaxis={'gridcolor': PALETTE['grid']},
                      margin=dict(t=50, b=30, l=160, r=80))
    return fig


def create_department_sentiment_bar(df: pd.DataFrame) -> go.Figure:
    """Grouped bar showing avg dissatisfaction per department."""
    if 'Department Name' not in df.columns or 'dissatisfaction_score' not in df.columns:
        return go.Figure()
    avg = (df.groupby('Department Name')['dissatisfaction_score']
           .mean().sort_values(ascending=False))
    colors = [PALETTE['danger'] if v >= 30 else PALETTE['warning']
              if v >= 15 else PALETTE['success'] for v in avg.values]
    fig = go.Figure(go.Bar(x=avg.index, y=avg.values,
                           marker_color=colors,
                           text=avg.values.round(1), textposition='outside',
                           textfont={'color': PALETTE['text']}))
    fig.update_layout(title='Avg Dissatisfaction by Department',
                      paper_bgcolor=PALETTE['card'], plot_bgcolor=PALETTE['bg'],
                      font={'color': PALETTE['text']}, height=300,
                      yaxis={'gridcolor': PALETTE['grid']},
                      margin=dict(t=50, b=60, l=40, r=20))
    return fig


def create_scatter_compound_vs_rating(df: pd.DataFrame) -> go.Figure:
    """Scatter plot: VADER compound score vs star rating."""
    if 'compound' not in df.columns:
        return go.Figure()
    # 'Rating' is optional — uploaded CSVs may not have it
    if 'Rating' not in df.columns:
        fig = go.Figure()
        fig.update_layout(title='Rating column not available in uploaded dataset',
                          paper_bgcolor=PALETTE['card'], font={'color': PALETTE['text']}, height=350)
        return fig
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig = px.scatter(sample, x='Rating', y='compound',
                     color='sentiment_class',
                     color_discrete_map=SENTIMENT_COLORS,
                     opacity=0.7, size_max=8,
                     labels={'compound': 'VADER Compound Score'},
                     hover_data=['Department Name'] if 'Department Name' in df.columns else None)
    fig.update_layout(title='VADER Compound Score vs Star Rating',
                      paper_bgcolor=PALETTE['card'], plot_bgcolor=PALETTE['bg'],
                      font={'color': PALETTE['text']}, height=350,
                      xaxis={'gridcolor': PALETTE['grid']},
                      yaxis={'gridcolor': PALETTE['grid']},
                      margin=dict(t=50, b=40, l=60, r=20))
    return fig


def create_wordcloud_image(texts: pd.Series, title: str = 'Most Frequent Words',
                           max_words: int = 150, bg_color: str = '#FFFFFF') -> bytes:
    """Generate a word-cloud and return it as PNG bytes."""
    combined = ' '.join(texts.dropna().astype(str).tolist())
    wc = WordCloud(
        width=900, height=400, background_color=bg_color,
        max_words=max_words, colormap='RdYlBu_r',
        collocations=False, min_font_size=10,
    ).generate(combined)
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=bg_color)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, color='#1E293B', fontsize=14, pad=10, fontweight='bold')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=bg_color)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def create_sarcasm_donut(df: pd.DataFrame) -> go.Figure:
    """Donut chart: sarcastic vs non-sarcastic reviews."""
    if 'is_sarcastic' not in df.columns:
        return go.Figure()
    counts = df['is_sarcastic'].value_counts()
    labels = ['Non-Sarcastic', 'Sarcastic / Ironic']
    values = [counts.get(False, 0), counts.get(True, 0)]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55,
                           marker_colors=[PALETTE['success'], PALETTE['danger']],
                           textinfo='label+percent',
                           textfont_color=PALETTE['text']))
    fig.update_layout(title='Sarcasm / Irony Detection Results',
                      paper_bgcolor=PALETTE['card'], height=300,
                      font={'color': PALETTE['text']},
                      margin=dict(t=50, b=10, l=10, r=10), showlegend=True)
    return fig


def create_irony_prob_histogram(df: pd.DataFrame) -> go.Figure:
    """Histogram of RoBERTa irony probability scores."""
    if 'irony_prob' not in df.columns:
        return go.Figure()
    fig = px.histogram(df, x='irony_prob', nbins=30,
                       color_discrete_sequence=[PALETTE['purple']],
                       labels={'irony_prob': 'Irony Probability (RoBERTa)'})
    fig.add_vline(x=0.55, line_dash='dash', line_color='#DC2626',
                  annotation_text='Threshold (0.55)', annotation_position='top right',
                  annotation_font_color='#DC2626')
    fig.update_layout(title='RoBERTa Irony Probability Distribution',
                      paper_bgcolor=PALETTE['card'], plot_bgcolor=PALETTE['bg'],
                      font={'color': PALETTE['text']}, height=300,
                      xaxis={'gridcolor': PALETTE['grid']},
                      yaxis={'gridcolor': PALETTE['grid']},
                      margin=dict(t=50, b=30, l=40, r=20))
    return fig


def create_age_sentiment_box(df: pd.DataFrame) -> go.Figure:
    """Box plot of dissatisfaction scores across age groups."""
    if 'Age' not in df.columns or 'dissatisfaction_score' not in df.columns:
        return go.Figure()
    df2 = df.copy()
    df2['Age Group'] = pd.cut(df2['Age'], bins=[0, 25, 35, 45, 55, 100],
                               labels=['<25', '25-35', '35-45', '45-55', '55+'])
    fig = px.box(df2, x='Age Group', y='dissatisfaction_score',
                 color='Age Group',
                 color_discrete_sequence=['#818CF8','#34D399','#FBBF24','#F87171','#60A5FA'])
    fig.update_layout(title='Dissatisfaction by Age Group',
                      paper_bgcolor=PALETTE['card'], plot_bgcolor=PALETTE['bg'],
                      font={'color': PALETTE['text']}, height=320,
                      yaxis={'gridcolor': PALETTE['grid']},
                      showlegend=False, margin=dict(t=50, b=40, l=60, r=20))
    return fig


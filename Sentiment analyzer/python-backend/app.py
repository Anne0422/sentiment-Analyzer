from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

# A simple "Topic Mapper" (Later you can replace this with your pre-trained LDA model)
def get_topic(text):
    text = text.lower()
    if any(w in text for w in ['size', 'fit', 'small', 'large']): return "Sizing Issue"
    if any(w in text for w in ['ship', 'delivery', 'late', 'arrive']): return "Logistics"
    if any(w in text for w in ['price', 'expensive', 'cost', 'money']): return "Pricing"
    return "General Feedback"

@app.route('/analyze-comment', methods=['POST'])
def analyze():
    data = request.get_json()
    comment_text = data.get('text', '')

    # 1. Sentiment Analysis
    scores = analyzer.polarity_scores(comment_text)
    sentiment = "Negative" if scores['compound'] <= -0.05 else "Positive"
    
    # 2. Topic Categorization
    topic = get_topic(comment_text)

    return jsonify({
        'status': 'success',
        'sentiment': sentiment,
        'topic': topic,
        'compound_score': scores['compound']
    })

if __name__ == '__main__':
    app.run(port=5000) # Runs on http://127.0.0.1:5000

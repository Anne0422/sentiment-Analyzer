from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    
    # Perform Sentiment Analysis
    vs = analyzer.polarity_scores(text)
    score = vs['compound']
    
    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return jsonify({
        "status": "success",
        "score": score,
        "label": label
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
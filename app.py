from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

app = Flask(__name__)

# Initialize the analyzer
analyzer = SentimentIntensityAnalyzer()

# Sentiment analysis function
def analyze_sentiment(review):
    # Clean the text
    review = str(review).lower()

    # Get polarity and subjectivity (Optional, if you want to display)
    polarity = TextBlob(review).sentiment.polarity
    subjectivity = TextBlob(review).sentiment.subjectivity

    # Get Vader sentiment score
    score = analyzer.polarity_scores(review)
    neg = score['neg']
    pos = score['pos']

    if neg > pos:
        sentiment = "Negative"
    elif pos > neg:
        sentiment = "Positive"
    else:
        sentiment = "Neutral"

    return sentiment, polarity, subjectivity

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Result page
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['review']
        sentiment, polarity, subjectivity = analyze_sentiment(review)

        return render_template('index.html', 
                               review=review, 
                               sentiment=sentiment,
                               polarity=polarity,
                               subjectivity=subjectivity)

if __name__ == '__main__':
    app.run(debug=True)

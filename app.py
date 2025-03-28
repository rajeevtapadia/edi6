from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline
import string
import pandas as pd
import datetime

app = Flask(__name__)

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize the summarizer and sentiment analysis pipelines
summarizer = pipeline("summarization")
sentiment_pipeline = pipeline("sentiment-analysis")
emotion_pipeline = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Sample data (replace with actual data fetching later)
data = {
    "id": range(1, 9),
    "text": [
        """Mahakumbh 2025: The grand spiritual gathering of Mahakumbh Mela is set to take place in Prayagraj, India, from January to April 2025. This once-in-12-years event attracts millions of devotees, sadhus, and tourists from around the world.""",
        """Coldplay India Concert 2024: Global music sensation Coldplay has announced their much-awaited concert in Mumbai as part of their 'Music of the Spheres' World Tour.""",
        """India vs Pakistan Cricket Match 2024: Cricket fans are gearing up for the high-voltage India vs Pakistan clash in the ICC T20 World Cup 2024.""",
        """Chandrayaan-4 Mission Announcement: The Indian Space Research Organisation (ISRO) has officially announced the launch of Chandrayaan-4, the next lunar exploration mission.""",
        """Ganesh Chaturthi 2024 Celebrations: The vibrant festival of Ganesh Chaturthi is set to begin in September 2024, with Mumbai and Pune preparing for grand processions.""",
        """Indian Budget 2025 Presentation: The Indian government is set to unveil the Union Budget 2025 in February, focusing on economic growth and digital infrastructure.""",
        """New Delhi AI Summit 2024: India will host the New Delhi AI Summit in October 2024, bringing together global tech leaders and policymakers.""",
        """Ayodhya Ram Temple Inauguration: The grand inauguration of the Ram Temple in Ayodhya is scheduled for January 2024, marking a historic moment."""
    ],
    "timestamp": [datetime.datetime.now() - datetime.timedelta(minutes=i*10) for i in range(8)]
}

df = pd.DataFrame(data)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = [word for word in word_tokenize(text) if word not in stop_words]
    return " ".join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '').lower()
    
    # Filter events based on query
    matching_events = df[df['text'].str.lower().str.contains(query)]
    
    results = []
    for _, event in matching_events.iterrows():
        # Generate summary
        summary = summarizer(event['text'], max_length=50, min_length=25, do_sample=False)[0]['summary_text']
        
        # Get sentiment
        sentiment = sentiment_pipeline(event['text'])[0]['label']
        
        # Get emotion
        emotion = emotion_pipeline(event['text'])[0]['label']
        
        results.append({
            'id': event['id'],
            'text': event['text'],
            'summary': summary,
            'sentiment': sentiment,
            'emotion': emotion,
            'timestamp': event['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True) 
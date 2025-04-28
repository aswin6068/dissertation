import streamlit as st
import pandas as pd
import base64
import os
import time
import shutil
import joblib
import spacy
import matplotlib.pyplot as plt
from gtts import gTTS
from pathlib import Path
from groq import Groq
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from collections import Counter
import ast
from newspaper import Article
from googleapiclient.discovery import build

# Initialize
st.set_page_config(page_title="Vision Assistant", layout="centered")
nlp = spacy.load("en_core_web_sm")

# Constants
groq_api_key = "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"
API_KEY = "AIzaSyAZBNUbKmVeHhRAaLmiyOVZ1GKfxBd56Xk"
CSE_ID = "d7eb56ceb292e4479"
dataset_file = "image_descriptions.csv"
dataset_path = os.path.join(os.getcwd(), dataset_file)

# Helper functions
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def convert_text_to_speech(text):
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)
    filename = os.path.join(output_folder, f"audio_{int(time.time())}.mp3")
    tts = gTTS(text=text[:2000], lang='en')
    tts.save(filename)
    return filename

def generate_vision_explanation(image_bytes):
    client = Groq(api_key=groq_api_key)
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Describe the image like explaining to a normal person."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}
        ],
        temperature=1,
        max_completion_tokens=500,
        stream=False
    )
    return completion.choices[0].message.content

def detect_obstacles(description):
    obstacle_keywords = ["barrier", "block", "obstruction", "pothole", "traffic", "closed", "fallen tree"]
    for keyword in obstacle_keywords:
        if keyword in description.lower():
            return "⚠️ Obstacle detected: {}".format(keyword)
    return "No major obstacles detected."

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words("english")]
    return " ".join(tokens)

def analyze_sentiment(text):
    score = TextBlob(text).sentiment.polarity
    return "Positive" if score > 0 else ("Negative" if score < 0 else "Neutral")

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def extract_product_name(description):
    entities = extract_entities(description)
    for text, label in entities:
        if label in ["PRODUCT", "WORK_OF_ART", "PERSON", "ORG"]:
            return text
    tokens = preprocess_text(description).split()
    return tokens[0] if tokens else "unknown"

def google_search(query, api_key, cse_id, num=5):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, num=num).execute()
    return [item['link'] for item in res.get('items', [])]

def fetch_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return ""

def get_product_feedback(product_query):
    links = google_search(f"{product_query} reviews", API_KEY, CSE_ID)
    sentiments = []
    positives, negatives = [], []
    for url in links:
        text = fetch_article(url)
        if text and len(text) > 150:
            sentiment = analyze_sentiment(text)
            if sentiment == "Positive":
                positives.append((text[:300], url))
            elif sentiment == "Negative":
                negatives.append((text[:300], url))
            sentiments.append(sentiment)
    if sentiments:
        total = len(sentiments)
        pos_count = sentiments.count("Positive")
        rating = round((pos_count / total) * 5, 1)
        feedback = f"⭐ Overall Rating: {rating}/5\n"
        if positives:
            feedback += f"+ Positive: {positives[0][0]}... Source: {positives[0][1]}\n"
        if negatives:
            feedback += f"- Negative: {negatives[0][0]}... Source: {negatives[0][1]}\n"
        return feedback
    else:
        return "No reviews found."

def save_to_csv(image_name, description):
    cleaned = preprocess_text(description)
    sentiment = analyze_sentiment(cleaned)
    entities = extract_entities(description)
    timestamp = pd.Timestamp.now()
    category = classify_text(cleaned)
    new_row = pd.DataFrame([{
        "image": image_name,
        "description": description,
        "cleaned_description": cleaned,
        "sentiment": sentiment,
        "named_entities": entities,
        "category": category,
        "timestamp": timestamp
    }])
    if os.path.exists(dataset_file):
        df = pd.read_csv(dataset_file)
        combined = pd.concat([df, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(dataset_file, index=False)

def train_text_classifier():
    if not os.path.exists(dataset_path):
        return
    dataset = pd.read_csv(dataset_path)
    if len(dataset) < 2:
        return
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["cleaned_description"],
        dataset["category"],
        test_size=0.2,
        random_state=42
    )
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(X_train, y_train)
    joblib.dump(model, "text_classifier.pkl")

def classify_text(description):
    if not os.path.exists("text_classifier.pkl"):
        return "Unknown"
    model = joblib.load("text_classifier.pkl")
    return model.predict([description])[0]

# --- Streamlit UI ---

st.title("🖼️ Vision-based Assistant")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    image_bytes = uploaded_image.read()
    if st.button("Generate Description and Feedback"):
        with st.spinner("Analyzing Image..."):
            description = generate_vision_explanation(image_bytes)
            obstacle_info = detect_obstacles(description)
            product_name = extract_product_name(description)
            feedback = get_product_feedback(product_name)
            final_message = f"Description: {description}\n{obstacle_info}\nFeedback: {feedback}"
            st.success("Analysis Complete!")
            st.write(final_message)
            save_to_csv(uploaded_image.name, description)
            train_text_classifier()
            tts_file = convert_text_to_speech(final_message)
            audio_file = open(tts_file, 'rb')
            st.audio(audio_file.read(), format='audio/mp3')

if os.path.exists(dataset_path):
    st.subheader("📊 Dataset Overview")
    df = pd.read_csv(dataset_path)
    st.dataframe(df)

    st.subheader("📈 Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts()
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    st.subheader("🔎 Top Entities")
    entity_lists = df["named_entities"].dropna().apply(ast.literal_eval)
    flat_entities = [ent[0].lower() for sublist in entity_lists for ent in sublist]
    entity_counter = Counter(flat_entities)
    st.write(entity_counter.most_common(10))

st.caption("Made with ❤️ using Streamlit + Groq + Google APIs")

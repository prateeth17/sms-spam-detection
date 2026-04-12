# ============================================================
# SMS SPAM DETECTION - STREAMLIT APP
# ============================================================
import os
import io
import nltk
import string
import requests
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

# ── Text preprocessing ───────────────────────────────────────
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

# ── Load spam.csv (local file or fallback to download) ───────
def get_dataframe():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "spam.csv")

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, encoding='latin1')

    # Fallback: download from UCI ML repo
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    response = requests.get(url)
    df = pd.read_csv(io.StringIO(response.text), sep='\t', header=None, names=['target', 'text'])
    return df

# ── Train model once and cache it ───────────────────────────
@st.cache_resource
def load_model():
    df = get_dataframe()

    # Handle both column formats
    if 'v1' in df.columns:
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
    for col in ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    encoder = LabelEncoder()
    df['target'] = encoder.fit_transform(df['target'])
    df.drop_duplicates(inplace=True)

    df['transformed_text'] = df['text'].apply(transform_text)

    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['transformed_text']).toarray()
    y = df['target'].values

    model = MultinomialNB()
    model.fit(X, y)

    return tfidf, model

# ── UI ───────────────────────────────────────────────────────
st.set_page_config(page_title="SMS Spam Detector")
st.title("📩 SMS Spam Detection System")

with st.spinner("Loading model..."):
    tfidf, model = load_model()

input_sms = st.text_area("Enter your message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message first.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam Message")

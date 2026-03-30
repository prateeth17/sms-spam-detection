# ============================================================
# SMS SPAM DETECTION - STREAMLIT APP
# ============================================================

import nltk
import string
import pickle
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# ── Text preprocessing ───────────────────────────────────────
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)

# ── Load model ───────────────────────────────────────────────
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# ── UI ───────────────────────────────────────────────────────
st.set_page_config(page_title="SMS Spam Detector")

st.title("📩 SMS Spam Detection System")

input_sms = st.text_area("Enter your message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam Message")

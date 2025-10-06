import re
import string
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

def extract_keywords(text: str) -> set:
    """
    Extract keywords from text: lowercase + remove stopwords & punctuation
    """
    text = clean_text(text)
    return set(text.split())

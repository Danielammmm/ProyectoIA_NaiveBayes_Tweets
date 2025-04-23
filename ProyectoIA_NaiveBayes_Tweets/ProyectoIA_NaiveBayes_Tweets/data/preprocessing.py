
import re
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import nltk

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Diccionario de expresiones y reducción para tokens informales
slang_map = {
    "omg": "surprise", "lol": "laugh", "lmao": "laugh", "rofl": "laugh",
    "bruh": "annoyed", "idk": "confused", "smh": "disappointed", "wtf": "anger",
    "ffs": "frustrated", "yass": "cheer", "yaaaas": "cheer", "noooo": "no",
    "yay": "cheer", "hahaha": "laugh", "bye": "neutral", "hi": "neutral",
    "hey": "neutral"
}

def reduce_repeated_letters(word):
    return re.sub(r'(.)\1{2,}', r'\1', word)

def normalize_slang(token):
    reduced = reduce_repeated_letters(token)
    return slang_map.get(reduced, reduced)

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Eliminar emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Unir negaciones
    text = re.sub(r"\bnot\s+(\w+)", r"not_\1", text)

    # Tokenizar, normalizar, quitar stopwords
    tokens = word_tokenize(text)
    normalized = [normalize_slang(t) for t in tokens]
    filtered = [w for w in normalized if w not in stop_words]

    return " ".join(filtered)

def tokenize(text):
    return word_tokenize(text.lower())

def build_bow(dataset):
    vocab = Counter()
    for text in dataset['Cleaned_Content']:
        tokens = tokenize(text)
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        all_tokens = tokens + bigrams
        vocab.update(all_tokens)
    return vocab

def text_to_bow(text, vocabulary):
    tokens = tokenize(text)
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
    all_tokens = tokens + bigrams

    bow_vector = [0] * len(vocabulary)
    vocab_index = {word: i for i, word in enumerate(vocabulary)}
    for token in all_tokens:
        if token in vocab_index:
            bow_vector[vocab_index[token]] += 1
    return bow_vector

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

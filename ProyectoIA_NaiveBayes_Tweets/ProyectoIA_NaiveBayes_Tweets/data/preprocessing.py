import pandas as pd
import nltk
import re
import pickle
from nltk.tokenize import word_tokenize
from collections import Counter

nltk.download('punkt')

# Cargar datasets .pkl
def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Tokenizador básico
def tokenize(text):
    return word_tokenize(text.lower())

# Construir bolsa de palabras de un dataset
def build_bow(dataset):
    all_tokens = []
    for text in dataset['Cleaned_Content']:
        tokens = tokenize(text)
        all_tokens.extend(tokens)
    return Counter(all_tokens)

# Convertir texto a representación BoW
def text_to_bow(text, vocabulary):
    tokens = tokenize(text)
    bow_vector = [0] * len(vocabulary)
    vocab_index = {word: i for i, word in enumerate(vocabulary)}
    for token in tokens:
        if token in vocab_index:
            bow_vector[vocab_index[token]] += 1
    return bow_vector

# Reutilizar para inferencia
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

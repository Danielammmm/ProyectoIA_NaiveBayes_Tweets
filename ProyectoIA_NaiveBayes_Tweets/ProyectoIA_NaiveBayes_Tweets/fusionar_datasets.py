
import pandas as pd
import os
import pickle
from data import preprocessing

# === 1. Cargar Sentiment140 ===
sent140_path = "data/training.1600000.processed.noemoticon.csv"
sent140_cols = ["target", "id", "date", "query", "user", "text"]
df_sent140 = pd.read_csv(sent140_path, encoding="ISO-8859-1", names=sent140_cols)

# Filtrar y mapear etiquetas
sentiment_map_140 = {0: "negative", 2: "neutral", 4: "positive"}
df_sent140 = df_sent140[df_sent140["target"].isin(sentiment_map_140.keys())]
df_sent140["Sentiment"] = df_sent140["target"].map(sentiment_map_140)
df_sent140 = df_sent140[["text", "Sentiment"]]
df_sent140.rename(columns={"text": "Content"}, inplace=True)

# === 2. Cargar Twitter Kaggle ===
kaggle_path = "data/Twitter_Tweets.csv"
df_kaggle = pd.read_csv(kaggle_path)

# Usar la columna 'text' como contenido
df_kaggle = df_kaggle[["text", "sentiment"]]
df_kaggle.columns = ["Content", "Sentiment"]

# Normalizar etiquetas
df_kaggle["Sentiment"] = df_kaggle["Sentiment"].str.lower()
df_kaggle = df_kaggle[df_kaggle["Sentiment"].isin(["positive", "negative", "neutral"])]

# === 3. Unificar datasets ===
df_combined = pd.concat([df_sent140, df_kaggle], ignore_index=True)

# === 4. Limpiar texto con tu preprocesamiento profesional ===
df_combined["Cleaned_Content"] = df_combined["Content"].apply(preprocessing.clean_text)
df_combined = df_combined[df_combined["Cleaned_Content"].str.strip() != ""]

# === 5. Guardar como cleaned_train.pkl ===
os.makedirs("data", exist_ok=True)
with open("data/cleaned_train.pkl", "wb") as f:
    pickle.dump(df_combined[["Cleaned_Content", "Sentiment"]], f)

print(f"✅ Dataset combinado limpiado y guardado con {len(df_combined)} registros.")

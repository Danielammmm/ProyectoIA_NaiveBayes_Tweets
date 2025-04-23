import pandas as pd
import pickle
import os
from data import preprocessing

# Cargar el dataset original
df = pd.read_csv("data/twitter_training.csv", header=None, names=["ID", "User", "Sentiment", "Content"])

# Filtrar clases válidas
df = df[df["Sentiment"].isin(["Positive", "Negative", "Neutral"])]

# Normalizar etiquetas
df["Sentiment"] = df["Sentiment"].str.lower()

# Limpiar el contenido con tu función mejorada
df["Cleaned_Content"] = df["Content"].apply(preprocessing.clean_text)

# Eliminar filas vacías tras limpieza
df = df[df["Cleaned_Content"].str.strip() != ""]

# Crear carpeta si no existe
os.makedirs("data", exist_ok=True)

# Guardar en formato .pkl
with open("data/cleaned_train.pkl", "wb") as f:
    pickle.dump(df[["Cleaned_Content", "Sentiment"]], f)

print("✅ cleaned_train.pkl generado exitosamente con", len(df), "registros.")

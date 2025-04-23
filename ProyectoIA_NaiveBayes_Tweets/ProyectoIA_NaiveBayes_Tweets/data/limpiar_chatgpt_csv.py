
import pandas as pd
import pickle
import os
import preprocessing

# Cargar el dataset generado por ChatGPT
df = pd.read_csv("chatgpt_training.csv")

# Normalizar etiquetas (opcional: tricky lo mantenemos)
df["Sentiment"] = df["Sentiment"].str.lower()

# Limpiar el texto con nuestro preprocesamiento quirúrgico
df["Cleaned_Content"] = df["Content"].apply(preprocessing.clean_text)

# Eliminar filas vacías después de limpiar
df = df[df["Cleaned_Content"].str.strip() != ""]

# Crear carpeta si no existe
os.makedirs("data", exist_ok=True)

# Guardar como cleaned_train.pkl
with open("data/cleaned_train.pkl", "wb") as f:
    pickle.dump(df[["Cleaned_Content", "Sentiment"]], f)

print(f"✅ cleaned_train.pkl generado con {len(df)} registros.")

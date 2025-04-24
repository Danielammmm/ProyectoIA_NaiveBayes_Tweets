# ProyectoIA_NaiveBayes_Tweets

Este proyecto consiste en una aplicación web que clasifica el sentimiento (positivo, negativo o neutro) de tweets usando el algoritmo Naïve Bayes implementado desde cero.

El sistema está desarrollado bajo una arquitectura en capas y contiene una interfaz web funcional para ingresar el texto a analizar y visualizar el resultado.
# Naïve Bayes Sentiment Classifier (Twitter-based)

Este proyecto es un clasificador de sentimientos entrenado con los datasets **Sentiment140** y **Twitter Tweets Sentiment (Kaggle)**, utilizando un modelo **Naïve Bayes personalizado desde cero**.

---

## Estructura del proyecto

```
IA_Tweets/
├── data/
│   ├── training.1600000.processed.noemoticon.csv
│   ├── Twitter_Tweets.csv
│   ├── cleaned_train.pkl
├── model/
│   ├── naive_bayes.py
│   └── model.pkl
├── service/
│   ├── app.py
│   └── ... (archivos de la API Flask)
├── web/
│   ├── templates/index.html
│   └── static/script.js, styles.css
├── preprocessing.py
├── fusionar_datasets.py
├── main.py
└── venv/
```

---

## Instalación y ejecución

### A. Usando entorno virtual (recomendado)

1. Crear y activar entorno virtual:
```bash
cd C:\IA_Tweets
python -m venv venv
venv\Scripts\activate
```

2. Instalar dependencias:
```bash
pip install pandas nltk matplotlib seaborn flask scikit-learn
```

3. Descargar recursos de NLTK:
```bash
python
```
```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
exit()
```

4. Preparar los datos:
```bash
python fusionar_datasets.py
```

5. Entrenar el modelo:
```bash
python main.py
```
NOTA: main.py puede tardar entre 30 a 40 minutos en trenar

6. Servir la API:
```bash
cd service
python app.py
```

Abrir navegador en:
```
http://127.0.0.1:5000
```

---

### B. Sin entorno virtual (limpiar instalación anterior)

1. Eliminar entorno virtual si existe:
```bash
rmdir /s /q venv
```

2. Instalar dependencias directamente en el sistema (no recomendado):
```bash
pip install pandas nltk matplotlib seaborn flask scikit-learn
```

3. Ejecutar los mismos pasos del punto A desde el paso 3.

---

## Extras

- El preprocesamiento incluye limpieza profunda, detección de lenguaje informal, bigramas y corrección de sarcasmo.
- El clasificador Naïve Bayes ha sido optimizado para trabajar con texto corto, sarcasmo, y estilo real de redes sociales.

---

## Dataset combinados

- [Sentiment140](http://help.sentiment140.com/for-students/)
- [Twitter Tweets Kaggle](https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset)




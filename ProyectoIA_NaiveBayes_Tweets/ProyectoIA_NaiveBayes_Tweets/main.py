"""import pickle
from data import preprocessing
from model.naive_bayes import NaiveBayesClassifier
from sklearn.metrics import accuracy_score
from model import metrics


# 1. Cargar datos
train_data = preprocessing.load_data('data/cleaned_train.pkl')
test_data = preprocessing.load_data('data/cleaned_test.pkl')

# 2. Crear vocabulario y convertir a BoW
bow_counts = preprocessing.build_bow(train_data)
vocab = list(bow_counts.keys())

X_train = [preprocessing.text_to_bow(text, vocab) for text in train_data['Cleaned_Content']]
y_train = train_data['Sentiment'].tolist()

X_test = [preprocessing.text_to_bow(text, vocab) for text in test_data['Cleaned_Content']]
y_test = test_data['Sentiment'].tolist()

# 3. Entrenar modelo
classifier = NaiveBayesClassifier()
classifier.train(X_train, y_train)

# 4. Predecir y evaluar
predictions = [classifier.predict(x) for x in X_test]
accuracy = accuracy_score(y_test, predictions)
print(f' Accuracy del modelo: {accuracy:.4f}')

metrics.evaluar_modelo(y_test, predictions)

# 5. Guardar modelo entrenado
import os
output_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
with open(output_path, 'wb') as f:
    pickle.dump((classifier, vocab), f)

print("Modelo guardado exitosamente en:", output_path)

"""
import pickle
from data import preprocessing
from model.naive_bayes import NaiveBayesClassifier
from sklearn.metrics import accuracy_score
from model import metrics
import os

# 1. Cargar datos
train_data = preprocessing.load_data('data/cleaned_train.pkl')
test_data = preprocessing.load_data('data/cleaned_test.pkl')

# 2. Crear vocabulario y convertir a BoW
bow_counts = preprocessing.build_bow(train_data)
vocab = list(bow_counts.keys())

X_train = [preprocessing.text_to_bow(text, vocab) for text in train_data['Cleaned_Content']]
y_train = train_data['Sentiment'].tolist()

X_test = [preprocessing.text_to_bow(text, vocab) for text in test_data['Cleaned_Content']]
y_test = test_data['Sentiment'].tolist()

# 3. Entrenar modelo
classifier = NaiveBayesClassifier()
classifier.train(X_train, y_train)

# 4. Predecir y evaluar
predictions = [classifier.predict(x) for x in X_test]
accuracy = accuracy_score(y_test, predictions)
print(f' Accuracy del modelo: {accuracy:.4f}')

# 5. Guardar modelo entrenado primero
output_path = os.path.join(os.path.dirname(__file__), 'model', 'model.pkl')
with open(output_path, 'wb') as f:
    pickle.dump((classifier, vocab), f)

print("Modelo guardado exitosamente en:", output_path)

# 6. Mostrar métricas (opcional visual)
metrics.evaluar_modelo(y_test, predictions)

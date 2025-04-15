import math
from collections import defaultdict

# ✅ Función nombrada para reemplazar la lambda (pickle-friendly)
def nested_defaultdict():
    return defaultdict(int)

class NaiveBayesClassifier:
    def __init__(self):
        self.vocab = set()
        self.word_counts = defaultdict(nested_defaultdict)   # ✅ Cambiado
        self.class_counts = defaultdict(int)
        self.class_total_words = defaultdict(int)
        self.total_docs = 0

    def train(self, X, y):
        self.vocab = set()
        self.word_counts = defaultdict(nested_defaultdict)   # ✅ Aquí también
        self.class_counts = defaultdict(int)
        self.class_total_words = defaultdict(int)
        self.total_docs = len(X)

        for bow_vector, label in zip(X, y):
            self.class_counts[label] += 1
            for word_index, count in enumerate(bow_vector):
                if count > 0:
                    self.word_counts[label][word_index] += count
                    self.class_total_words[label] += count
                    self.vocab.add(word_index)

    def predict(self, bow_vector):
        best_label = None
        max_log_prob = float('-inf')

        for label in self.class_counts:
            log_prob = math.log(self.class_counts[label] / self.total_docs)

            for word_index, count in enumerate(bow_vector):
                if count > 0:
                    word_freq = self.word_counts[label].get(word_index, 0)
                    total_words = self.class_total_words[label]
                    vocab_size = len(self.vocab)
                    # Suavizado de Laplace
                    prob = (word_freq + 1) / (total_words + vocab_size)
                    log_prob += count * math.log(prob)

            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_label = label

        return best_label

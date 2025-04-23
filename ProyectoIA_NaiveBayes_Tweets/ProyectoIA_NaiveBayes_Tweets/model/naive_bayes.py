
import math
from collections import defaultdict

def nested_defaultdict():
    return defaultdict(int)

class NaiveBayesClassifier:
    def __init__(self):
        self.vocab = set()
        self.word_counts = defaultdict(nested_defaultdict)
        self.class_counts = defaultdict(int)
        self.class_total_words = defaultdict(int)
        self.total_docs = 0

    def train(self, X, y):
        self.vocab = set()
        self.word_counts = defaultdict(nested_defaultdict)
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
        probs = self.predict_proba(bow_vector)
        return max(probs, key=probs.get)

    def predict_proba(self, bow_vector):
        log_probs = {}
        word_contributions = {}

        for label in self.class_counts:
            log_prob = math.log(self.class_counts[label] / self.total_docs)
            contributions = []

            for word_index, count in enumerate(bow_vector):
                if count > 0:
                    word_freq = self.word_counts[label].get(word_index, 0)
                    total_words = self.class_total_words[label]
                    vocab_size = len(self.vocab)

                    # Laplace smoothing
                    prob = (word_freq + 1) / (total_words + vocab_size)
                    contribution = count * math.log(prob)
                    log_prob += contribution
                    contributions.append((word_index, contribution))

            log_probs[label] = log_prob
            word_contributions[label] = contributions

        max_log = max(log_probs.values())
        exp_probs = {k: math.exp(v - max_log) for k, v in log_probs.items()}
        total = sum(exp_probs.values())
        probs = {k: round(v / total, 4) for k, v in exp_probs.items()}

        self._last_word_contributions = word_contributions
        return probs

    def get_token_contributions(self, vocab_list):
        result = {}
        for label, contribs in self._last_word_contributions.items():
            sorted_contribs = sorted(contribs, key=lambda x: x[1], reverse=True)
            result[label] = [(vocab_list[i], round(score, 4)) for i, score in sorted_contribs if score != 0][:10]
        return result

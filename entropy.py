from collections import Counter
import math


def probability_distribution(array: list[str]):
    counter = Counter(array) # Hashmapa w ilością wystąpień elementów w tablicy

    total = sum(counter.values())

    for key in counter:
        counter[key] /= total

    return counter

def calculate_entropy(values):
    probabilities = probability_distribution(values)

    entropy = 0
    for key in probabilities:
        entropy -= probabilities[key] * math.log2(probabilities[key])

    return entropy
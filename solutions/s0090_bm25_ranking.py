from collections import Counter

import numpy as np


def calculate_bm25_scores(
    corpus: list[list[str]], query: list[str], k1: float = 1.5, b: float = 0.75
) -> np.ndarray:
    doc_lengths = [len(doc) for doc in corpus]
    avg_doc_length = np.mean(doc_lengths)
    doc_term_counts = [Counter(doc) for doc in corpus]
    doc_freqs = Counter()

    for doc in corpus:
        doc_freqs.update(set(doc))

    scores = np.zeros(len(corpus))

    for term in query:
        df = doc_freqs.get(term, 0) + 1
        idf = np.log((len(corpus) + 1) / df)

        for idx, term_counts in enumerate(doc_term_counts):
            if term not in term_counts:
                continue

            tf = term_counts[term]
            doc_len_norm = 1 - b + b * (doc_lengths[idx] / avg_doc_length)
            term_score = (tf * (k1 + 1)) / (tf + k1 * doc_len_norm)
            scores[idx] += idf * term_score

    return np.round(scores, 3)

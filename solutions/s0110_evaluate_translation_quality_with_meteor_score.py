from collections import Counter, defaultdict


def meteor_score(
    reference: str,
    candidate: str,
    alpha: float = 0.9,
    beta: float = 3,
    gamma: float = 0.5,
) -> float:
    ref_words = [word.lower() for word in reference.split()]
    cand_words = [word.lower() for word in candidate.split()]

    if not ref_words or not cand_words:
        return 0.0

    ref_counter = Counter(ref_words)
    cand_counter = Counter(cand_words)
    matches = sum(
        min(cand_count, ref_counter[word])
        for word, cand_count in cand_counter.items()
        if word in ref_counter
    )

    if matches == 0:
        return 0.0

    precision = matches / len(cand_words)
    recall = matches / len(ref_words)

    denominator = alpha * precision + (1 - alpha) * recall
    f_mean = (precision * recall) / denominator if denominator != 0 else 0.0

    word_positions = defaultdict(list)
    for pos, word in enumerate(ref_words):
        word_positions[word].append(pos)

    used = defaultdict(int)
    alignments = []
    for word in cand_words:
        if word in word_positions and used[word] < len(word_positions[word]):
            pos = word_positions[word][used[word]]
            alignments.append(pos)
            used[word] += 1
        else:
            alignments.append(None)

    chunks = 0
    prev_pos = -1
    for pos in alignments:
        if pos is not None:
            if prev_pos == -1 or pos != prev_pos + 1:
                chunks += 1
            prev_pos = pos
        else:
            prev_pos = -1

    penalty = gamma * (chunks / matches) ** beta
    score = f_mean * (1 - penalty)

    return round(score, 3)

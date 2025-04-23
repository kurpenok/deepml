def learn_decision_tree(
    examples: list[dict], attributes: list[str], target_attr: str
) -> dict:
    from math import log2

    def entropy(values):
        count = {}
        total = len(values)
        for v in values:
            count[v] = count.get(v, 0) + 1
        e = 0.0
        for c in count.values():
            p = c / total
            if p > 0:
                e -= p * log2(p)
        return e

    def information_gain(data, attr, target, current_entropy):
        splits = {}
        for ex in data:
            val = ex[attr]
            if val not in splits:
                splits[val] = []
            splits[val].append(ex)
        total = len(data)
        new_entropy = 0.0
        for subset in splits.values():
            subset_targets = [ex[target] for ex in subset]
            new_entropy += (len(subset) / total) * entropy(subset_targets)
        return current_entropy - new_entropy

    def majority_class(values):
        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        return max(counts, key=lambda k: counts[k])

    target_values = [ex[target_attr] for ex in examples]
    unique_targets = set(target_values)

    if len(unique_targets) == 1:
        return unique_targets.pop()

    if not attributes:
        return majority_class(target_values)

    current_entropy = entropy(target_values)
    best_gain = -1
    best_attr = None

    for attr in attributes:
        gain = information_gain(examples, attr, target_attr, current_entropy)
        if gain > best_gain:
            best_gain, best_attr = gain, attr

    splits = {}
    for ex in examples:
        val = ex[best_attr]
        if val not in splits:
            splits[val] = []
        splits[val].append(ex)

    remaining_attrs = [a for a in attributes if a != best_attr]
    tree = {best_attr: {}}

    for value, subset in splits.items():
        subset_targets = [ex[target_attr] for ex in subset]
        unique_sub = set(subset_targets)
        if len(unique_sub) == 1:
            tree[best_attr][value] = unique_sub.pop()
        else:
            subtree = learn_decision_tree(subset, remaining_attrs, target_attr)
            tree[best_attr][value] = subtree

    return tree

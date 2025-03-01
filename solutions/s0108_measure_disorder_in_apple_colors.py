from collections import Counter


def disorder(apples: list) -> float:
    return 1 - sum((count / len(apples)) ** 2 for count in Counter(apples).values())

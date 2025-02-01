def gini_impurity(y: list[int]) -> float:
    classes = set(y)

    gini_impurity = 0

    for cls in classes:
        gini_impurity += (y.count(cls) / len(y)) ** 2

    return round(1 - gini_impurity, 3)

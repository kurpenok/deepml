def calculate_brightness(img: list[list[int]]) -> float:
    if (
        img
        and all(len(img[0]) == len(row) for row in img)
        and all(0 <= p <= 255 for row in img for p in row)
    ):
        return sum(p for row in img for p in row) / (len(img) * len(img[0]))
    return -1

def calculate_brightness(img: list[list[int]]) -> float:
    if not img:
        return -1

    m = len(img)
    n = len(img[0])

    s = 0
    for row in img:
        if len(row) != n:
            return -1

        for pixel in row:
            if not (0 <= pixel <= 255):
                return -1
            s += pixel

    return s / (n * m)

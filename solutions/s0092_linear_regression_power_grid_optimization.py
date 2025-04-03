import math


def power_grid_forecast(consumption_data: list[int]) -> int:
    days = list(range(1, 11))
    n = len(days)

    cleaned = []
    for i in range(n):
        day = days[i]
        fluctuation = 10 * math.sin(2 * math.pi * day / 10)
        cleaned.append(consumption_data[i] - fluctuation)

    sum_x = sum(days)
    sum_y = sum(cleaned)
    sum_xy = sum(x * y for x, y in zip(days, cleaned))
    sum_x2 = sum(x**2 for x in days)

    m_numerator = n * sum_xy - sum_x * sum_y
    m_denominator = n * sum_x2 - sum_x**2
    m = m_numerator / m_denominator
    b = (sum_y - m * sum_x) / n

    base_15 = m * 15 + b
    fluctuation_15 = 10 * math.sin(2 * math.pi * 15 / 10)
    pred_15 = base_15 + fluctuation_15

    rounded_pred = round(pred_15)
    final_prediction = round(rounded_pred * 1.05)

    return final_prediction + 1

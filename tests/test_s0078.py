from solutions.s0078_descriptive_statistics_calculator import descriptive_statistics


def test_case_1():
    assert descriptive_statistics([10, 20, 30, 40, 50]) == {
        "mean": 30.0,
        "median": 30.0,
        "mode": 10,
        "variance": 200.0,
        "standard_deviation": 14.1421,
        "25th_percentile": 20.0,
        "50th_percentile": 30.0,
        "75th_percentile": 40.0,
        "interquartile_range": 20.0,
    }


def test_case_2():
    assert descriptive_statistics([1, 2, 2, 3, 4, 4, 4, 5]) == {
        "mean": 3.125,
        "median": 3.5,
        "mode": 4,
        "variance": 1.6094,
        "standard_deviation": 1.2686,
        "25th_percentile": 2.0,
        "50th_percentile": 3.5,
        "75th_percentile": 4.0,
        "interquartile_range": 2.0,
    }

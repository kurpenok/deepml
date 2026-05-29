from src.s0066_implement_orthogonal_projection_of_a_vector_onto_a_line import (
    orthogonal_projection,
)


def test_case_1():
    v = [3.0, 4.0]
    L = [1.0, 0.0]

    result = orthogonal_projection(v, L)
    expected = [3.0, 0.0]

    assert result == expected


def test_case_2():
    v = [1.0, 2.0, 3.0]
    L = [0.0, 0.0, 1.0]

    result = orthogonal_projection(v, L)
    expected = [0.0, 0.0, 3.0]

    assert result == expected


def test_case_3():
    v = [5.0, 6.0, 7.0]
    L = [2.0, 0.0, 0.0]

    result = orthogonal_projection(v, L)
    expected = [5.0, 0.0, 0.0]

    assert result == expected

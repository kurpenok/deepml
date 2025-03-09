import numpy as np

from solutions.s0019_principal_component_analysis_implementation import pca


def test_case_1():
    assert (
        pca(np.array([[1, 2], [3, 4], [5, 6]]), 1) == np.array([[0.7071], [0.7071]])
    ).all()


def test_case_2():
    assert (
        pca(np.array([[4, 2, 1], [5, 6, 7], [9, 12, 1], [4, 6, 7]]), 2)
        == np.array([[0.6855, 0.0776], [0.6202, 0.4586], [-0.3814, 0.8853]])
    ).all()

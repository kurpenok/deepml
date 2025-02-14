import numpy as np

from solutions.s0082_grayscale_image_contrast_calculator import calculate_contrast


def test_case_1():
    assert calculate_contrast(np.array([[0, 50], [200, 255]])) == 255


def test_case_2():
    assert calculate_contrast(np.array([[128, 128], [128, 128]])) == 0

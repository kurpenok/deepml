from solutions.s0101_implement_the_grpo_objective_function import grpo_objective


def test_case_1():
    assert (
        grpo_objective(
            [1.2, 0.8, 1.1],
            [1.0, 1.0, 1.0],
            [0.9, 1.1, 1.0],
            [1.0, 0.5, 1.5],
            epsilon=0.2,
            beta=0.01,
        )
        == 1.032749
    )


def test_case_2():
    assert (
        grpo_objective(
            [0.9, 1.1], [1.0, 1.0], [1.0, 1.0], [0.8, 1.2], epsilon=0.1, beta=0.05
        )
        == 0.999743
    )


def test_case_3():
    assert (
        grpo_objective(
            [1.5, 0.5, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.2, 0.7, 1.3],
            epsilon=0.15,
            beta=0.02,
        )
        == 0.882682
    )


def test_case_4():
    assert grpo_objective([1.0], [1.0], [1.0], [1.0], epsilon=0.1, beta=0.01) == 1.0

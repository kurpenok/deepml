from solutions.s0060_implement_tf_idf import compute_tf_idf


def test_case_1():
    assert compute_tf_idf(
        [
            ["the", "cat", "sat", "on", "the", "mat"],
            ["the", "dog", "chased", "the", "cat"],
            ["the", "bird", "flew", "over", "the", "mat"],
        ],
        ["cat"],
    ) == [[0.21461], [0.25754], [0.0]]


def test_case_2():
    assert compute_tf_idf(
        [
            ["the", "cat", "sat", "on", "the", "mat"],
            ["the", "dog", "chased", "the", "cat"],
            ["the", "bird", "flew", "over", "the", "mat"],
        ],
        ["cat", "mat"],
    ) == [[0.21461, 0.21461], [0.25754, 0.0], [0.0, 0.21461]]


def test_case_3():
    assert compute_tf_idf(
        [
            ["this", "is", "a", "sample"],
            ["this", "is", "another", "example"],
            ["yet", "another", "sample", "document"],
            ["one", "more", "document", "for", "testing"],
        ],
        ["sample", "document", "test"],
    ) == [
        [0.37771, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.37771, 0.37771, 0.0],
        [0.0, 0.30217, 0.0],
    ]

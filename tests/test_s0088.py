from solutions.s0088_gpt_2_text_generation import gen_text


def test_case_1():
    assert gen_text("hello", 5) == "hello hello hello <UNK> <UNK>"


def test_case_2():
    assert (
        gen_text("hello world", n_tokens_to_generate=10)
        == "world world world world world world world world world world"
    )


def test_case_3():
    assert gen_text("world", n_tokens_to_generate=3) == "world world world"

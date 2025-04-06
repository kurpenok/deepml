from solutions.s0110_evaluate_translation_quality_with_meteor_score import meteor_score


def test_case_1():
    assert (
        meteor_score("Rain falls gently from the sky", "Gentle rain drops from the sky")
        == 0.625
    )


def test_case_2():
    assert (
        meteor_score("The dog barks at the moon", "The dog barks at the moon") == 0.998
    )


def test_case_3():
    assert meteor_score("The sun shines brightly", "Clouds cover the sky") == 0.125


def test_case_4():
    assert meteor_score("Birds sing in the trees", "Birds in the trees sing") == 0.892

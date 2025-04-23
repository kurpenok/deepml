from solutions.s0020_decision_tree_learing import learn_decision_tree


def test_case_1():
    assert learn_decision_tree(
        [
            {"Outlook": "Sunny", "Wind": "Weak", "PlayTennis": "No"},
            {"Outlook": "Overcast", "Wind": "Strong", "PlayTennis": "Yes"},
            {"Outlook": "Rain", "Wind": "Weak", "PlayTennis": "Yes"},
            {"Outlook": "Sunny", "Wind": "Strong", "PlayTennis": "No"},
            {"Outlook": "Sunny", "Wind": "Weak", "PlayTennis": "Yes"},
            {"Outlook": "Overcast", "Wind": "Weak", "PlayTennis": "Yes"},
            {"Outlook": "Rain", "Wind": "Strong", "PlayTennis": "No"},
            {"Outlook": "Rain", "Wind": "Weak", "PlayTennis": "Yes"},
        ],
        ["Outlook", "Wind"],
        "PlayTennis",
    ) == {
        "Outlook": {
            "Sunny": {"Wind": {"Weak": "No", "Strong": "No"}},
            "Rain": {"Wind": {"Weak": "Yes", "Strong": "No"}},
            "Overcast": "Yes",
        }
    }

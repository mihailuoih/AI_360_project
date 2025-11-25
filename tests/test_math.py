from AgentBasedModel.utils import math as math_utils


def test_rolling_handles_none_and_returns_expected_length():
    data = [1, None, 3, 4]
    res = math_utils.rolling(data, 2)
    assert len(res) == 3
    assert res[0] == 1
    assert res[1] == 3
    assert res[2] == 3.5


def test_difference_computes_pairwise():
    assert math_utils.difference([1, 4, 9]) == [3, 5]

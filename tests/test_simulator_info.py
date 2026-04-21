import math

from AgentBasedModel.agents.agents import ExchangeAgent
from AgentBasedModel.simulator.simulator import SimulatorInfo


def _empty_info():
    return SimulatorInfo(ExchangeAgent(), [])


def test_stock_returns_handles_zero_prices_without_crash():
    info = _empty_info()
    info.prices = [100.0, 0.0, 100.0, 110.0]
    info.dividends = [0.0, 0.0, 0.0, 0.0]

    returns = info.stock_returns(1)

    assert len(returns) == 3
    assert all(math.isfinite(x) for x in returns)
    assert returns[1] == 0.0


def test_liquidity_ignores_invalid_points_in_mean_mode():
    info = _empty_info()
    info.prices = [100.0, 0.0, 90.0]
    info.spreads = [{"bid": 99.0, "ask": 101.0}, None, {"bid": 89.0, "ask": 91.0}]

    liq_series = info.liquidity(1)
    liq_mean = info.liquidity()

    assert liq_series[1] is None
    assert liq_series[0] == 0.02
    assert round(liq_series[2], 6) == round(2.0 / 90.0, 6)
    expected_mean = (liq_series[0] + liq_series[2]) / 2
    assert round(liq_mean, 10) == round(expected_mean, 10)

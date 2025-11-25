import math

from AgentBasedModel.agents.agents import ExchangeAgent
from AgentBasedModel.simulator import SimulatorInfo
from AgentBasedModel.states.states import detect_shock_end, panic


def _build_info(prices):
    exchange = ExchangeAgent()
    info = SimulatorInfo(exchange, [])
    info.prices = prices
    info.dividends = [0 for _ in prices]
    return info


def test_panic_detects_volatility_spike():
    calm = [100 + 0.1 * math.sin(i) for i in range(80)]
    turbulent = []
    price = calm[-1]
    for i in range(80):
        price += (-1) ** i * (5 + i % 3)
        turbulent.append(price)
    prices = calm + turbulent
    info = _build_info(prices)
    assert panic(info, size=None, window=5, th=1.5, baseline_window=40, consec=2)


def test_panic_remains_false_when_series_stable():
    prices = [100 + 0.01 * i for i in range(120)]
    info = _build_info(prices)
    assert not panic(info, size=None, window=5, th=2.0, baseline_window=50, consec=3)


def test_detect_shock_end_identifies_recovery():
    pre = [100.0] * 40
    shock = [80.0] * 5
    recovery = [90 + i * 0.5 for i in range(20)]
    stable = [100.0] * 40
    prices = pre + shock + recovery + stable
    t0 = len(pre)
    info = _build_info(prices)
    res = detect_shock_end(
        info,
        t0=t0,
        W_ref=20,
        W_stab=10,
        consec_ok=2,
        conf=0.9,
        slope_th=5e-3,
        vol_relax=2.0,
        band_k_sigma=3.0,
        ema_alpha=0.3,
        max_horizon=200,
    )
    assert res["t_end"] is not None
    assert res["t_end"] > t0
    assert math.isclose(res["p_star"], 100.0, abs_tol=1.0)

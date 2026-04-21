import importlib.util
from pathlib import Path

import pytest

from AgentBasedModel.agents.agents import ExchangeAgent, HFMarketMaker, Random


SPEC = importlib.util.spec_from_file_location("main_module", Path(__file__).resolve().parents[1] / "main.py")
main = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(main)


def test_maker_endowment_reads_mode_specific_config():
    cfg = {
        "agent_cash": 1000,
        "agent_assets": 0,
        "market_maker_endowment": {
            "mm": {"cash": 500, "assets": 5},
            "hfmm": {"cash": 750, "assets": 7},
        },
    }

    assert main.maker_endowment(cfg, "mm") == (500.0, 5)
    assert main.maker_endowment(cfg, "hfmm") == (750.0, 7)
    assert main.maker_endowment(cfg, "amm") == (1000.0, 0)


def test_build_traders_keeps_retail_and_maker_endowments_separate():
    exchange = ExchangeAgent(price=100, std=0, volume=0)
    traders = main.build_traders(
        exchange=exchange,
        agents_cfg={"Random": 2},
        mode="hfmm",
        agent_cash=1000,
        agent_assets=0,
        maker_cash=500,
        maker_assets=5,
    )

    retail = [trader for trader in traders if isinstance(trader, Random)]
    makers = [trader for trader in traders if isinstance(trader, HFMarketMaker)]

    assert len(retail) == 2
    assert all(trader.cash == 1000 for trader in retail)
    assert all(trader.assets == 0 for trader in retail)

    assert len(makers) == 1
    assert makers[0].cash == 500
    assert makers[0].assets == 5


def test_validate_maker_endowment_rejects_zero_reserves_for_hfmm():
    cfg = {
        "exchange": {"price": 100},
        "market_maker_endowment": {"hfmm": {"cash": 1000, "assets": 0}},
    }

    with pytest.raises(ValueError, match="HFMM requires positive maker reserves"):
        main.validate_maker_endowment(cfg, "hfmm")


def test_validate_maker_endowment_rejects_wrong_price_ratio_for_amm():
    cfg = {
        "exchange": {"price": 100},
        "market_maker_endowment": {"amm": {"cash": 500, "assets": 10}},
    }

    with pytest.raises(ValueError, match="AMM initial reserve ratio must match exchange price"):
        main.validate_maker_endowment(cfg, "amm")


def test_validate_maker_endowment_ignores_mm():
    cfg = {
        "exchange": {"price": 100},
        "market_maker_endowment": {"mm": {"cash": 1, "assets": 0}},
    }

    main.validate_maker_endowment(cfg, "mm")

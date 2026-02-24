import json
import random
import argparse
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from AgentBasedModel.agents.agents import (
    AutoMarketMaker,
    Chartist,
    ExchangeAgent,
    Fundamentalist,
    HFMarketMaker,
    MarketMaker,
    Random,
    Universalist,
)
from AgentBasedModel.events.events import FundamentalPriceShockRelative, AgentExitShock
from AgentBasedModel.simulator import Simulator
from AgentBasedModel.states import aggToShock
from AgentBasedModel.states.states import detect_shock_end, detect_spread_recovery, panic
from AgentBasedModel.visualization.market import (
    plot_liquidity,
    plot_price,
    plot_price_fundamental,
    plot_volatility_price,
    plot_volatility_return,
)

CONFIG_PATH = Path("config/scenarios.json")
RESULTS_PATH = Path("results/summary.csv")
PLOTS_DIR = Path("results/plots")

AGENT_FACTORIES = {
    "Random": Random,
    "Fundamentalist": Fundamentalist,
    "Chartist": Chartist,
    "Universalist": Universalist,
}

PLOT_REGISTRY = {
    "price": plot_price,
    "price_fundamental": plot_price_fundamental,
    "price_volatility": plot_volatility_price,
    "return_volatility": plot_volatility_return,
    "liquidity": plot_liquidity,
}

SHOCK_PARAM_KEYS = [
    "max_horizon",
    "W_ref",
    "W_stab",
    "consec_ok",
    "conf",
    "slope_th",
    "vol_relax",
    "band_k_sigma",
    "ema_alpha",
]


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def resolve_config_path() -> Path:
    """
    Priority:
    1) CLI --config <path>
    2) ENV ABM_CONFIG
    3) default CONFIG_PATH
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON")
    args, _ = parser.parse_known_args()
    if args.config:
        return Path(args.config)
    env_cfg = os.environ.get("ABM_CONFIG")
    if env_cfg:
        return Path(env_cfg)
    return CONFIG_PATH


def default_agent_presets() -> List[Dict]:
    return [
        {
            "id": "baseline",
            "description": "Равномерный состав трейдеров",
            "agents": {"Random": 10, "Fundamentalist": 10, "Chartist": 10, "Universalist": 10},
        }
    ]


def price_levels(cfg: Dict) -> List[int]:
    if not cfg:
        return list(range(1, 11)) + list(range(15, 91, 5))
    min_percent = int(cfg.get("min_percent", 1))
    max_percent = int(cfg.get("max_percent", 90))
    if max_percent <= 0:
        return [0]
    switch = int(cfg.get("switch_percent", 10))
    fine_step = int(cfg.get("fine_step", 1))
    coarse_step = int(cfg.get("coarse_step", 5))

    fine = list(range(min_percent, min(max_percent, switch) + 1, max(1, fine_step)))
    coarse_start = max(switch + coarse_step, min_percent)
    coarse = list(range(coarse_start, max_percent + 1, max(1, coarse_step)))
    levels = sorted(set(fine + coarse))
    return [value for value in levels if 0 < value <= max_percent]


def liquidity_levels(cfg: Dict) -> List[float]:
    if not cfg:
        return [round(x / 10, 1) for x in range(1, 10)]
    if "fractions" in cfg:
        return [float(x) for x in cfg["fractions"] if 0 < float(x) <= 1]
    min_fraction = float(cfg.get("min_fraction", 0.1))
    max_fraction = float(cfg.get("max_fraction", 0.9))
    step = float(cfg.get("step", 0.1))
    levels = []
    current = min_fraction
    while current <= max_fraction + 1e-9:
        levels.append(round(current, 3))
        current += step
    return levels


def build_exchange(cfg: Dict) -> ExchangeAgent:
    exchange_cfg = cfg.get("exchange", {})
    return ExchangeAgent(
        price=exchange_cfg.get("price", 100),
        std=exchange_cfg.get("std", 25),
        volume=exchange_cfg.get("volume", 1000),
        rf=exchange_cfg.get("rf", 5e-4),
        transaction_cost=exchange_cfg.get("transaction_cost", 0),
    )


def build_traders(exchange: ExchangeAgent, agents_cfg: Dict, mode: str, cash: float, assets: int) -> List:
    traders = []
    for name, count in agents_cfg.items():
        factory = AGENT_FACTORIES.get(name)
        if factory is None:
            continue
        for _ in range(int(count)):
            traders.append(factory(exchange, cash, assets))

    if mode == "mm":
        traders.append(MarketMaker(exchange, cash, assets))
    elif mode == "amm":
        traders.append(AutoMarketMaker(exchange, cash, assets))
    elif mode == "hfmm":
        traders.append(HFMarketMaker(exchange, cash, assets))
    else:
        raise ValueError(f"Unknown market mode: {mode}")
    return traders


def instantiate_events(specs: List[Dict]) -> List:
    return [spec["cls"](**spec["params"]) for spec in specs]


def generate_plots(info, scenario_id: str, plot_cfgs: List[Dict], enabled: bool = True) -> List[str]:
    if not enabled:
        return []
    saved = []
    if not plot_cfgs:
        return saved
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for idx, plot_cfg in enumerate(plot_cfgs):
        plot_type = plot_cfg.get("type")
        func = PLOT_REGISTRY.get(plot_type)
        if func is None:
            continue
        kwargs = dict(plot_cfg.get("kwargs", {}))
        if "figsize" in kwargs and isinstance(kwargs["figsize"], list):
            kwargs["figsize"] = tuple(kwargs["figsize"])
        filename = f"{scenario_id}_{plot_type}_{idx:02d}.png"
        out_path = PLOTS_DIR / filename
        kwargs["save_path"] = out_path
        func(info, **kwargs)
        saved.append(str(out_path))
    return saved


def run_simulation(cfg: Dict, preset: Dict, mode: str, scenario_meta: Dict, event_specs: List[Dict], seed: int) -> Dict:
    random.seed(seed)

    exchange = build_exchange(cfg)
    agent_cash = cfg.get("agent_cash", 10 ** 3)
    agent_assets = cfg.get("agent_assets", 0)
    traders = build_traders(exchange, preset["agents"], mode, agent_cash, agent_assets)
    events = instantiate_events(event_specs)

    iterations = cfg.get("iterations", 600)
    simulator = Simulator(exchange=exchange, traders=traders, events=events)
    simulator.simulate(iterations, silent=True)

    info = simulator.info

    panic_cfg = cfg.get("panic", {})
    panic_flag = panic(
        info,
        size=None,
        window=panic_cfg.get("short_window", 10),
        th=panic_cfg.get("threshold", 2.0),
        baseline_window=panic_cfg.get("baseline_window", 50),
        consec=panic_cfg.get("consec", 3),
    )

    plot_files = generate_plots(
        info,
        scenario_meta["scenario_run_id"],
        cfg.get("plots", []),
        enabled=cfg.get("save_plots", True),
    )

    detect_cfg = cfg.get("shock_detection", {})
    detect_kwargs = {k: detect_cfg[k] for k in SHOCK_PARAM_KEYS if k in detect_cfg}
    t0 = scenario_meta["t0"]
    detect_kwargs["t0"] = t0

    event_label, event_obj, price_diag = extract_event_diagnostics(simulator)

    spread_rec = detect_spread_recovery(
        info,
        t0=t0,
        W_ref=detect_kwargs.get("W_ref", 50),
        W_stab=detect_kwargs.get("W_stab", 30),
        consec_ok=detect_kwargs.get("consec_ok", 3),
        relax=detect_cfg.get("spread_relax", detect_cfg.get("vol_relax", 1.5)),
        band_k_sigma=detect_cfg.get("band_k_sigma", 2.0),
    )

    row = build_result_row(
        info=info,
        panic_flag=panic_flag,
        plot_files=plot_files,
        detect_kwargs=detect_kwargs,
        event_label=event_label,
        event=event_obj,
        price_diag=price_diag,
        scenario_meta=scenario_meta,
        agent_counts=preset["agents"],
        simulator=simulator,
    )
    row["t_spread_end"] = spread_rec.get("t_spread_end")
    row["spread_ref"] = spread_rec.get("spread_ref")
    row["spread_post"] = spread_rec.get("spread_post")
    row["spread_duration"] = (row["t_spread_end"] - t0) if (row.get("t_spread_end") is not None and t0 is not None) else None
    return row


def extract_event_diagnostics(simulator: Simulator):
    events = simulator.events or []
    label = None
    event = None
    price_diag = {}
    if events:
        event = events[0]
        label = str(event)
        agg = aggToShock(simulator, 1, [("price", lambda data, _: data.prices)])
        price_diag = agg.get(label, {}).get("price", {})
    return label, event, price_diag


def build_result_row(
    info,
    panic_flag: bool,
    plot_files: List[str],
    detect_kwargs: Dict,
    event_label: str,
    event,
    price_diag: Dict,
    scenario_meta: Dict,
    agent_counts: Dict,
    simulator: Simulator,
) -> Dict:
    t0 = detect_kwargs.get("t0")
    detect_params = {k: v for k, v in detect_kwargs.items() if k != "t0"}
    shock_stats = detect_shock_end(info, t0=t0, **detect_params) if t0 is not None else {}
    t_end = shock_stats.get("t_end")
    prices = info.prices
    p_before = prices[t0 - 1] if (t0 is not None and t0 - 1 < len(prices)) else None
    p_at_t_end = prices[t_end] if (t_end is not None and t_end < len(prices)) else None

    row = {
        "scenario_id": scenario_meta["scenario_id"],
        "scenario_run_id": scenario_meta["scenario_run_id"],
        "scenario_type": scenario_meta["scenario_type"],
        "agent_preset": scenario_meta["agent_preset"],
        "mode": scenario_meta["mode"],
        "shock_value": scenario_meta["shock_value"],
        "shock_value_kind": scenario_meta["shock_value_kind"],
        "repeat_idx": scenario_meta["repeat_idx"],
        "seed": scenario_meta["seed"],
        "panic_detected": bool(panic_flag),
        "shock_event": event_label or "N/A",
        "t_end": t_end,
        "shock_duration": (t_end - t0) if (t_end is not None and t0 is not None) else None,
        "p_star": shock_stats.get("p_star"),
        "p_before": p_before,
        "p_at_t_end": p_at_t_end,
        "p_right_before": price_diag.get("right before"),
        "p_right_after": price_diag.get("right after"),
        "plot_files": ";".join(plot_files),
    }

    def _mean_clean(seq):
        seq = [x for x in seq if x is not None]
        return sum(seq) / len(seq) if seq else None

    def _std_clean(seq):
        seq = [x for x in seq if x is not None]
        if len(seq) < 2:
            return None
        m = sum(seq) / len(seq)
        return (sum((x - m) ** 2 for x in seq) / len(seq)) ** 0.5

    row["spread_mean"] = _mean_clean(info.spread_sizes)
    row["spread_std"] = _std_clean(info.spread_sizes)
    row["rel_spread_mean"] = _mean_clean(info.rel_spreads)
    row["rel_spread_std"] = _std_clean(info.rel_spreads)
    row["top_qty_mean"] = _mean_clean(info.top_qty)
    row["top_qty_std"] = _std_clean(info.top_qty)
    row["depth_band_mean"] = _mean_clean(info.depth_band)
    row["depth_band_std"] = _std_clean(info.depth_band)
    row["spread_per_vol_mean"] = _mean_clean(info.spread_per_volume)
    row["spread_per_vol_std"] = _std_clean(info.spread_per_volume)
    row["n_orders_bid_mean"] = _mean_clean([c["bid"] for c in info.order_counts])
    row["n_orders_ask_mean"] = _mean_clean([c["ask"] for c in info.order_counts])
    row["n_orders_bid_std"] = _std_clean([c["bid"] for c in info.order_counts])
    row["n_orders_ask_std"] = _std_clean([c["ask"] for c in info.order_counts])

    for name, factory in AGENT_FACTORIES.items():
        row[f"n_{name.lower()}"] = agent_counts.get(name, 0)
    row["n_marketmaker"] = 1 if scenario_meta["mode"] == "mm" else 0
    row["n_automarketmaker"] = 1 if scenario_meta["mode"] == "amm" else 0
    row["n_hfmm"] = 1 if scenario_meta["mode"] == "hfmm" else 0
    return row


def main(config_path: Path = None):
    cfg_path = config_path or CONFIG_PATH
    cfg = load_config(cfg_path)
    presets = cfg.get("agent_presets") or default_agent_presets()
    price_cfg = cfg.get("price_shock", {})
    liquidity_cfg = cfg.get("liquidity_shock", {})
    base_seed = int(cfg.get("base_seed", 42))
    repeats = int(cfg.get("repeats", 1))
    price_t0 = int(price_cfg.get("t0", 200))
    liquidity_t0 = int(liquidity_cfg.get("t0", 250))

    modes = cfg.get("modes") or ["mm", "amm", "hfmm"]
    price_values = price_levels(price_cfg)
    liquidity_values = liquidity_levels(liquidity_cfg)

    tasks = []
    for preset in presets:
        preset_id = preset.get("id", "preset")
        for mode in modes:
            for percent in price_values:
                # если шок = 0, шоковое событие не добавляем (базовый сценарий)
                price_events = []
                scenario_type = "price"
                if percent > 0:
                    price_events = [
                        {
                            "cls": FundamentalPriceShockRelative,
                            "params": {"it": price_t0, "fraction": -percent / 100.0},
                        }
                    ]
                else:
                    scenario_type = "base"
                tasks.append(
                    {
                        "scenario_type": scenario_type,
                        "preset": preset,
                        "mode": mode,
                        "value": percent,
                        "value_kind": "percent",
                        "t0": price_t0,
                        "event_specs": price_events,
                        "scenario_id": f"price_{preset_id}_{mode}_{percent:02d}",
                    }
                )
            for fraction in liquidity_values:
                percent_value = int(round(fraction * 100))
                tasks.append(
                    {
                        "scenario_type": "liquidity",
                        "preset": preset,
                        "mode": mode,
                        "value": fraction,
                        "value_kind": "fraction",
                        "t0": liquidity_t0,
                        "event_specs": [
                            {
                                "cls": AgentExitShock,
                                "params": {"it": liquidity_t0, "fraction": fraction},
                            }
                        ],
                        "scenario_id": f"liquidity_{preset_id}_{mode}_{percent_value:02d}",
                    }
                )

    rows = []
    scenario_counter = 0

    for idx, task in enumerate(tqdm(tasks, desc="Scenarios", total=len(tasks)), start=1):
        print(
            f"[{idx}/{len(tasks)}] {task['scenario_id']} ({task['scenario_type']} shock {task['value']}, mode={task['mode']})",
            flush=True,
        )
        scenario_counter = run_repeated(
            cfg=cfg,
            preset=task["preset"],
            mode=task["mode"],
            scenario_type=task["scenario_type"],
            scenario_id=task["scenario_id"],
            event_specs=task["event_specs"],
            shock_value=task["value"],
            shock_value_kind=task["value_kind"],
            t0=task["t0"],
            repeats=repeats,
            base_seed=base_seed,
            scenario_counter=scenario_counter,
            rows=rows,
        )

    df = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_PATH, index=False)
    print(df)


def run_repeated(
    cfg: Dict,
    preset: Dict,
    mode: str,
    scenario_type: str,
    scenario_id: str,
    event_specs: List[Dict],
    shock_value,
    shock_value_kind: str,
    t0: int,
    repeats: int,
    base_seed: int,
    scenario_counter: int,
    rows: List[Dict],
) -> int:
    show_progress = cfg.get("show_repeats_progress", True)
    iterator = range(1, repeats + 1)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Repeats {scenario_id}", leave=False)

    for repeat_idx in iterator:
        seed = base_seed + scenario_counter * repeats + repeat_idx
        scenario_meta = {
            "scenario_id": scenario_id,
            "scenario_run_id": f"{scenario_id}_run{repeat_idx}",
            "scenario_type": scenario_type,
            "agent_preset": preset.get("id", "preset"),
            "mode": mode,
            "shock_value": shock_value,
            "shock_value_kind": shock_value_kind,
            "repeat_idx": repeat_idx,
            "seed": seed,
            "t0": t0,
        }
        row = run_simulation(cfg, preset, mode, scenario_meta, event_specs, seed)
        rows.append(row)
    return scenario_counter + 1


if __name__ == "__main__":
    main(resolve_config_path())

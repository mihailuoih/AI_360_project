from AgentBasedModel.simulator import Simulator, SimulatorInfo
import AgentBasedModel.utils.math as math

from scipy.stats import kendalltau
import statsmodels.api as sm


def aggToShock(sim: Simulator, window: int, funcs: list) -> dict:
    """
    Aggregate market statistics in respect to market shocks

    :param sim: Simulator object
    :param funcs: [('func_name', func), ...]. Function accepts SimulatorInfo object and roll or window variable
    :param window: 1 to n
    :return:
    """
    if not sim.events:
        return {}
    result = {}
    for event in sim.events:
        event_stats = {}
        for f_name, f in funcs:
            series = f(sim.info, window)
            if not series:
                event_stats[f_name] = {
                    'start': None,
                    'before': [],
                    'right before': None,
                    'after': [],
                    'right after': None,
                    'end': None
                }
                continue
            anchor = event.it - window
            start_val = series[0]
            before_slice = series[:max(0, anchor)]
            right_before = series[anchor] if 0 <= anchor < len(series) else None
            after_start = anchor + 1
            after_slice = series[after_start:] if after_start < len(series) else []
            right_after = series[after_start] if 0 <= after_start < len(series) else None
            end_val = series[-1]
            event_stats[f_name] = {
                'start': start_val,
                'before': before_slice,
                'right before': right_before,
                'after': after_slice,
                'right after': right_after,
                'end': end_val
            }
        result[str(event)] = event_stats
    return result


def test_trend_kendall(values, category: bool = False, conf: float = .95) -> bool or dict:
    """
    Kendall’s Tau test.
    H0: No trend exists
    Ha: Some trend exists
    :return: True - trend exist, False - no trend
    """
    iterations = range(len(values))
    tau, p_value = kendalltau(iterations, values)
    if category:
        return p_value < (1 - conf)
    return {'tau': round(tau, 4), 'p-value': round(p_value, 4)}


def test_trend_ols(values) -> dict:
    """
    Linear regression on time.
    H0: No trend exists
    Ha: Some trend exists
    :return: True - trend exist, False - no trend
    """
    x = range(len(values))
    estimate = sm.OLS(values, sm.add_constant(x)).fit()
    return {
        'value': round(estimate.params[1], 4),
        't-stat': round(estimate.tvalues[1], 4),
        'p-value': round(estimate.pvalues[1], 4)
    }


def trend(info: SimulatorInfo, size: int = None, window: int = 5, conf: float = .95, th: float = .01) -> bool or list:
    prices = info.prices[window:]

    if size is None:
        test = test_trend_ols(prices)
        return test['p-value'] < (1 - conf) and abs(test['value']) > th

    res = list()
    for i in range(len(prices) // size):
        test = test_trend_ols(prices[i*size:(i+1)*size])
        res.append(test['p-value'] < (1 - conf) and abs(test['value']) > th)

    return res


def _rolling_std(values: list, window: int) -> list:
    if window <= 0 or len(values) < window:
        return []
    return [math.std(values[i-window:i]) for i in range(window, len(values) + 1)]


def _require_consecutive(flags: list, consec: int) -> list:
    if consec <= 1:
        return flags
    streak = 0
    res = []
    for flag in flags:
        streak = streak + 1 if flag else 0
        res.append(streak >= consec)
    return res


def panic(
    info: SimulatorInfo,
    size: int = None,
    window: int = 5,
    th: float = 2.0,
    baseline_window: int = 50,
    consec: int = 3
) -> bool or list:
    """
    Panic is detected when short-horizon volatility spikes relative to a longer baseline
    for several consecutive ticks. Uses return series to stay scale-invariant.
    """
    returns = info.stock_returns(1)
    if len(returns) < max(window, baseline_window):
        return False if size is None else [False] * (len(info.prices) // size)

    short_std = _rolling_std(returns, window)
    long_std = _rolling_std(returns, baseline_window)
    if not short_std or not long_std:
        return False if size is None else [False] * (len(info.prices) // size)

    min_len = min(len(short_std), len(long_std))
    short_std = short_std[-min_len:]
    long_std = long_std[-min_len:]
    ratios = []
    for s, l in zip(short_std, long_std):
        if l == 0:
            ratios.append(float('inf') if s > 0 else 0.0)
        else:
            ratios.append(s / l)
    panic_flags = _require_consecutive([r >= th for r in ratios], consec)

    per_tick_flags = [False] * len(info.prices)
    start_idx = baseline_window
    for offset, flag in enumerate(panic_flags):
        idx = start_idx + offset
        if idx < len(per_tick_flags):
            per_tick_flags[idx] = flag

    if size is None:
        return any(per_tick_flags)

    res = []
    n_segments = len(info.prices) // size
    for seg in range(n_segments):
        chunk = per_tick_flags[seg*size:(seg+1)*size]
        res.append(any(chunk))
    return res


def disaster(info: SimulatorInfo, size: int = None, window: int = 5, conf: float = .95, th: float = .02) -> bool or list:
    volatility = info.price_volatility(window)
    if size is None:
        test = test_trend_ols(volatility)
        return test['value'] > th and test['p-value'] < (1 - conf)

    res = list()
    for i in range(len(volatility) // size):
        test = test_trend_ols(volatility[i*size:(i+1)*size])
        res.append(test['value'] > th and test['p-value'] < (1 - conf))
    return res


def mean_rev(info: SimulatorInfo, size: int = None, window: int = 5, conf: float = .95, th: float = -.02) -> bool or list:
    volatility = info.price_volatility(window)
    if size is None:
        test = test_trend_ols(volatility)
        return test['value'] < th and test['p-value'] < (1 - conf)

    res = list()
    for i in range(len(volatility) // size):
        test = test_trend_ols(volatility[i*size:(i+1)*size])
        res.append(test['value'] < th and test['p-value'] < (1 - conf))
    return res


def general_states(info: SimulatorInfo, size: int = 10, window: int = 5) -> str or list:
    states_trend = trend(info, size)
    states_panic = panic(info, size, window)
    states_disaster = disaster(info, size, window)
    states_mean_rev = mean_rev(info, size, window)

    res = list()
    for t, p, d, mr in zip(states_trend, states_panic, states_disaster, states_mean_rev):
        if mr:
            res.append('mean-rev')
        elif d:
            res.append('disaster')
        elif p:
            res.append('panic')
        elif t:
            res.append('trend')
        else:
            res.append('stable')
    return res

# NOT TESTED CODE

def _ema(series, alpha: float = 0.2):
    """Простая EMA без внешних зависимостей."""
    if not series:
        return []
    out = [float(series[0])]
    for x in series[1:]:
        out.append(alpha * float(x) + (1 - alpha) * out[-1])
    return out


def _sub_trend_no_signal(values, conf: float, slope_th: float) -> bool:
    """
    True -> тренда НЕТ (устойчиво)
    False -> тренд ЕСТЬ (неустойчиво)
    Использует test_trend_ols.
    """
    if len(values) < 5:
        # слишком мало данных — не считаем это трендом
        return True
    res = test_trend_ols(values)
    slope = abs(res['value'])
    pval = res['p-value']
    # тренда нет, если либо статистически незначимо, либо наклон очень мал
    return (pval >= (1 - conf)) or (slope <= slope_th)


def detect_shock_end(
    info: SimulatorInfo,
    t0: int,
    *,
    max_horizon: int = 2000,      # как далеко после t0 ищем окончание
    W_ref: int = 50,              # окно "докризисной" базы до шока
    W_stab: int = 30,             # окно стабильности (для проверки условий)
    consec_ok: int = 3,           # сколько подряд тиков должны выполниться все условия
    conf: float = 0.95,           # доверие для теста тренда
    slope_th: float = 1e-2,       # порог для |наклона| тренда
    vol_relax: float = 1.25,      # допустимое отношение постшоковой волатильности к базовой
    band_k_sigma: float = 2.0,    # ширина ценового коридора в сигмах базы
    ema_alpha: float = 0.2        # если нет фундаментала, используем EMA как "норму"
) -> dict:
    """
    Определяет момент окончания шока и послешоковый уровень цены.

    :param info: SimulatorInfo с полями prices, dividends, spreads и т.п.
    :param t0:   момент шока (event.it)
    :return: {
        't_end': int | None,
        'p_star': float | None,
        'diagnostics': {...}
    }
    """
    prices = list(info.prices)
    n = len(prices)
    if n == 0 or t0 >= n - 1:
        return {'t_end': None, 'p_star': None, 'diagnostics': {'reason': 'not enough data', 't0': t0, 'n': n}}

    # --- 1) Докризисная база ---
    L_ref = max(0, t0 - W_ref)
    ref_slice = prices[L_ref:t0] if t0 > L_ref else prices[:max(1, t0)]
    if len(ref_slice) < 2:
        ref_slice = prices[:max(2, t0)]
    p_ref = float(sum(ref_slice) / len(ref_slice)) if ref_slice else float(prices[t0])
    try:
        sigma_price_ref = float(math.std(ref_slice))
    except Exception:
        sigma_price_ref = float((max(ref_slice) - min(ref_slice)) / 6.0) if len(ref_slice) >= 2 else 0.0

    # Базовый безопасный абсолютный бенд, если сигма крошечная
    abs_band_floor = 0.01 * (abs(p_ref) if p_ref != 0 else 1.0) # <-- THERE

    # --- 2) Нормальный уровень: fundamental_value() -> EMA ---
    p_norm_series = None
    # пробуем фундаментал, если метод существует и возвращает вектор длины n
    try:
        fv = info.fundamental_value()
        if fv is None:
            raise RuntimeError
        if hasattr(fv, '__len__') and len(fv) == n:
            p_norm_series = list(map(float, fv))
    except Exception:
        p_norm_series = None

    if p_norm_series is None:
        p_norm_series = _ema(prices, alpha=ema_alpha)
    else:
        # используем фундаментал до шока, далее обновляем EMA без заглядывания в будущее
        if len(p_norm_series) < n:
            p_norm_series.extend([p_norm_series[-1]] * (n - len(p_norm_series)))
        tail_start = min(max(t0, 0), n - 1)
        ema_state = p_norm_series[tail_start]
        for idx in range(tail_start + 1, n):
            ema_state = ema_alpha * float(prices[idx]) + (1 - ema_alpha) * ema_state
            p_norm_series[idx] = ema_state

    # --- 3) Проверка устойчивости на хвосте ---
    def is_stable_at(t: int) -> bool:
        if t < t0 + 1:
            return False
        L = max(0, t - W_stab + 1)
        R = t + 1
        wnd = prices[L:R]
        if len(wnd) < max(5, min(10, W_stab // 2)):
            return False

        # (a) тренда НЕТ
        no_trend = _sub_trend_no_signal(wnd, conf=conf, slope_th=slope_th)

        # (b) волатильность вернулась к базе
        try:
            sigma_tail = float(math.std(wnd))
        except Exception:
            sigma_tail = 0.0
        vol_ok = (sigma_price_ref == 0.0) or (sigma_tail <= vol_relax * sigma_price_ref)

        # (c) цена в коридоре нормы
        p_t = float(prices[t])
        p_norm_t = float(p_norm_series[t])
        band_sigma = band_k_sigma * sigma_price_ref
        band = max(band_sigma, abs_band_floor)
        band_ok = abs(p_t - p_norm_t) <= band

        return bool(no_trend and vol_ok and band_ok)

    ok = 0
    t_end = None
    stop_t = min(n - 1, t0 + max_horizon)
    for t in range(t0 + 1, stop_t + 1):
        if is_stable_at(t):
            ok += 1
            if ok >= consec_ok:
                t_end = t
                break
        else:
            ok = 0

    if t_end is None:
        return {
            't_end': None,
            'p_star': None,
            'diagnostics': {
                'reason': 'not found',
                't0': t0, 'n': n,
                'p_ref': p_ref,
                'sigma_price_ref': sigma_price_ref,
                'params': {
                    'W_ref': W_ref, 'W_stab': W_stab, 'consec_ok': consec_ok,
                    'conf': conf, 'slope_th': slope_th, 'vol_relax': vol_relax,
                    'band_k_sigma': band_k_sigma, 'ema_alpha': ema_alpha, 'max_horizon': max_horizon
                }
            }
        }

    # --- 4) Послешоковый уровень p* ---
    Lp = max(0, t_end - W_stab + 1)
    tail = prices[Lp:t_end + 1]
    p_star = float(sum(tail) / len(tail)) if tail else float(prices[t_end])

    return {
        't_end': int(t_end),
        'p_star': float(p_star),
        'diagnostics': {
            't0': t0,
            'p_ref': p_ref,
            'sigma_price_ref': sigma_price_ref,
            'p_norm_at_end': float(p_norm_series[t_end]),
            'vol_tail': float(math.std(tail)) if len(tail) >= 2 else 0.0,
            'params': {
                'W_ref': W_ref, 'W_stab': W_stab, 'consec_ok': consec_ok,
                'conf': conf, 'slope_th': slope_th, 'vol_relax': vol_relax,
                'band_k_sigma': band_k_sigma, 'ema_alpha': ema_alpha, 'max_horizon': max_horizon
            }
        }
    }


def detect_spread_recovery(
    info: SimulatorInfo,
    t0: int,
    *,
    W_ref: int = 50,
    W_stab: int = 30,
    consec_ok: int = 3,
    relax: float = 1.5,
    band_k_sigma: float = 2.0
) -> dict:
    """
    Момент восстановления ликвидности по спреду: средний спред возвращается в коридор базового уровня.
    """
    if not hasattr(info, "spread_sizes") or not info.spread_sizes:
        return {}
    spreads = info.spread_sizes
    if t0 is None or t0 <= W_ref or len(spreads) < t0 + W_stab:
        return {}

    base_slice = [s for s in spreads[max(0, t0 - W_ref):t0] if s is not None]
    if not base_slice:
        return {}
    mean_ref = sum(base_slice) / len(base_slice)
    sigma_ref = math.std(base_slice) if len(base_slice) > 1 else 0.0

    ok = 0
    t_spread_end = None
    for t in range(t0, min(len(spreads) - W_stab + 1, t0 + 5000)):
        wnd = [s for s in spreads[t:t + W_stab] if s is not None]
        if not wnd:
            ok = 0
            continue
        mean_w = sum(wnd) / len(wnd)
        cond_mean = mean_w <= mean_ref * relax
        cond_band = sigma_ref == 0.0 or mean_w <= mean_ref + band_k_sigma * sigma_ref
        if cond_mean and cond_band:
            ok += 1
            if ok >= consec_ok:
                t_spread_end = t + W_stab - 1
                break
        else:
            ok = 0

    return {
        "t_spread_end": t_spread_end,
        "spread_ref": mean_ref,
        "spread_post": mean_w if t_spread_end is not None else None,
    }

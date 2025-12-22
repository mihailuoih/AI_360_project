from math import exp
import numpy as np
from scipy import stats


def mean(x: list) -> float:

    return sum(x) / len(x)


def quantile(x: list, q=.5) -> float:
    return sorted(x)[round(len(x) * q) - 1]


def std(x: list) -> float:
    m = mean(x)
    return (sum([(i - m)**2 for i in x]) / len(x))**.5


def rolling(x: list, n) -> list:
    if n is None or n <= 0 or len(x) < n:
        return []

    if None not in x:
        return [mean(x[i:i+n]) for i in range(len(x) - n + 1)]

    res = list()
    for i in range(len(x) - n + 1):
        xs = [el for el in x[i:i+n] if el is not None]
        res.append(mean(xs) if xs else None)
    return res


def difference(x: list) -> list:
    return [x[i+1] - x[i] for i in range(len(x) - 1)]


def aggregate(types_arr: list, target_arr: list, labels) -> dict:
    data = {tr_str: list() for tr_str in labels}
    for it in range(len(target_arr)):
        tmp = {tr_str: list() for tr_str in labels}
        for tr_id in target_arr[it].keys():
            if types_arr[it][tr_id] in labels:
                tmp[types_arr[it][tr_id]].append(target_arr[it][tr_id])
        for k, v in tmp.items():
            if v:
                v = mean(v)
            else:
                v = None
            data[k].append(v)
    return data


def paired_ttest(sample_a, sample_b, alpha: float = 0.05) -> dict:
    """
    Парный t-тест: H0 о нулевой средней разности между sample_a и sample_b.
    Игнорирует None/NaN, требует >= 2 наблюдений.
    """
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask]
    b = b[mask]
    n = len(a)
    if n < 2:
        return {"n": n, "t_stat": None, "p_value": None, "mean_diff": None, "ci_low": None, "ci_high": None}

    diff = a - b
    mean_diff = float(np.mean(diff))
    t_stat, p_value = stats.ttest_rel(a, b, nan_policy="omit")

    se = stats.sem(diff, nan_policy="omit")
    if np.isnan(se) or se == 0:
        ci_low = ci_high = mean_diff
    else:
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin = t_crit * se
        ci_low = mean_diff - margin
        ci_high = mean_diff + margin

    return {
        "n": n,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": mean_diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }

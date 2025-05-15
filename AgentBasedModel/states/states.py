from AgentBasedModel.simulator import Simulator, SimulatorInfo
import AgentBasedModel.utils.math as math

from scipy.stats import kendalltau, chi2
import statsmodels.api as sm


def aggToShock(sim: Simulator, window: int, funcs: list) -> dict:
    """
    Aggregate market statistics in respect to market shocks

    :param sim: Simulator object
    :param funcs: [('func_name', func), ...]. Function accepts SimulatorInfo object and roll or window variable
    :param window: 1 to n
    :return:
    """
    return {str(event): {f_name: {
        'start': f(sim.info, window)[0],
        'before': f(sim.info, window)[:event.it - window] if event.it - window > 0 else [],
        'right before': f(sim.info, window)[event.it - window] if event.it - window - 1 >= 0 else [],
        'after': f(sim.info, window)[event.it - window + 1:],
        'right after': f(sim.info, window)[event.it - window + 1],
        'end': f(sim.info, window)[-1]
    } for f_name, f in funcs} for event in sim.events}


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


def panic(info: SimulatorInfo, size: int = None, window: int = 5, th: float = .5) -> bool or list:
    volatility = info.price_volatility(window)
    if size is None:
        return any(v > th for v in volatility)

    res = list()
    for i in range(len(volatility) // size):
        sl = volatility[i*size:(i+1)*size]
        res.append(any(v > math.mean(volatility) * th for v in sl))
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

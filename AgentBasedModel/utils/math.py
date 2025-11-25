from math import exp


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

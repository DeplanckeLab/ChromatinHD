import numpy as np
import pandas as pd


def digits2number(digits, base=2, fractional=False):
    powers = np.arange(len(digits))
    out = (digits * base**powers).sum()
    if fractional:
        out = out / base ** (len(digits))
    return out


def number2digits(n, base=2):
    if n == 0:
        return np.array()
    nDigits = np.ceil(np.log(n + 1) / np.log(base))
    powers = base ** (np.arange(nDigits + 1))
    out = np.diff(n % powers) / powers[:-1]
    return out


def vanDerCorput(n, base=2, start=1):
    out = np.array(
        [
            digits2number(number2digits(ii, base)[::-1], base, True)
            for ii in range(1, n + start)
        ]
    )
    return out


def offset(
    y,
    maxLength=None,
    method="quasirandom",
    nbins=20,
    adjust=1,
    bw_method="scott",
    max_density=None,
):
    if len(y) == 1:
        return [0]

    if isinstance(y, pd.Series):
        y = y.values

    if nbins is None:
        if method in ["pseudorandom", "quasirandom"]:
            nbins = 2**10
        else:
            nbins = int(max(2, np.ceil(len(y) / 5)))

    if maxLength is None:
        subgroup_width = 1
    else:
        subgroup_width = np.sqrt(len(y) / maxLength)

    #     from sklearn.neighbors import KernelDensity
    #     kde = KernelDensity(kernel='gaussian').fit(y.reshape(-1, 1))
    #     pointDensities = np.exp(np.array(kde.score_samples(y.reshape(-1, 1))))

    import scipy.stats

    kernel = scipy.stats.gaussian_kde(y, bw_method=bw_method)
    pointDensities = kernel(y)

    if max_density is None:
        max_density = pointDensities.max()

    pointDensities = pointDensities / max_density

    if method == "quasirandom":
        offset = np.array(vanDerCorput(len(y)))[np.argsort(y)]
    elif method == "pseudorandom":
        offset = np.random.uniform(size=len(y))

    out = (offset - 0.5) * 2 * pointDensities * subgroup_width * 0.5

    return out

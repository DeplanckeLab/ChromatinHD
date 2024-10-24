import matplotlib.ticker as ticker

import decimal

import functools
import math


def count_zeros(value):
    decimal_value = str(decimal.Decimal((value)))
    if "." in decimal_value:
        count = len(decimal_value.split(".")[1])
    else:
        count = 0
    return count


def round_significant(value, significant=2):
    """
    Round to a significant number of digits, including trailing zeros.
    For example round_significant(15400, 2) = 15000
    """

    if value == 0:
        return 0

    n = math.log10(abs(value))
    return round(value, significant - int(n) - 1)


def format_distance(value, tick_pos=None, add_sign=True):
    abs_value = int(abs(value))
    if abs_value >= 1000000:
        # zeros = 0
        zeros = len(str(abs_value).rstrip("0")) - 1
        abs_value = abs_value / 1000000
        suffix = "mb"
    elif abs_value >= 1000:
        zeros = 0
        # zeros = len(str(abs_value).rstrip("0")) - 1
        abs_value = abs_value / 1000
        suffix = "kb"
    elif abs_value == 0:
        # zeros = 0
        return "TSS"
    else:
        zeros = 0
        suffix = "b"

    formatted_value = ("{abs_value:." + str(zeros) + "f}{suffix}").format(abs_value=abs_value, suffix=suffix)

    if not add_sign:
        return formatted_value
    return f"-{formatted_value}" if value < 0 else f"+{formatted_value}"


gene_ticker = ticker.FuncFormatter(format_distance)


def custom_formatter(value, tick_pos, base=1):
    if base != 1:
        value = value / base

    abs_value = abs(value)
    if abs_value >= 1000000:
        abs_value = abs_value / 1000000
        suffix = "mb"
    elif abs_value >= 1000:
        abs_value = abs_value / 1000
        suffix = "kb"
    elif abs_value == 0:
        return "0"
    else:
        suffix = "b"

    zeros = count_zeros(abs_value)

    formatted_value = ("{abs_value:." + str(zeros) + "f}{suffix}").format(abs_value=abs_value, suffix=suffix)
    return formatted_value


distance_ticker = ticker.FuncFormatter(custom_formatter)


def DistanceFormatter(base=1):
    return ticker.FuncFormatter(functools.partial(custom_formatter, base=base))


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

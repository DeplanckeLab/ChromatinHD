def interval_contains_inclusive(x, y):
    """
    Determines whether the intervals in x are contained in any interval of y
    """
    contained = ~((y[:, 1] < x[:, 0][:, None]) | (y[:, 0] > x[:, 1][:, None]))
    return contained.any(1)

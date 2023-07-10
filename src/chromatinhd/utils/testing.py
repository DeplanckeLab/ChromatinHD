import numpy as np
import scipy.stats as stats


def autocorrelation(x):
    """
    Calculate the autocorrelation of a sequence.

    Args:
    x (list): A sequence of numbers.

    Returns:
    float: Autocorrelation value.
    """
    n = len(x)
    x_bar = np.mean(x)
    numerator = np.sum([(x[i] - x_bar) * (x[i - 1] - x_bar) for i in range(1, n)])
    denominator = np.sum([(x[i] - x_bar) ** 2 for i in range(n)])

    return numerator / denominator


def repeated_kfold_corrected_t_test(
    performance_A, performance_B, k, num_repeats, alpha=0.05
):
    """
    Perform the corrected t-test between two learning algorithms A and B for repeated K-fold cross-validation.

    Args:
    performance_A (list): A list of performance scores for algorithm A.
    performance_B (list): A list of performance scores for algorithm B.
    k (int): Number of folds in the cross-validation.
    num_repeats (int): Number of times the K-fold cross-validation was repeated.
    alpha (float): Significance level, default is 0.05.

    Returns:
    bool: True if there is a significant difference, False otherwise.
    float: t-statistic value.
    float: p-value.
    """

    if (
        len(performance_A) != len(performance_B)
        or len(performance_A) != k * num_repeats
    ):
        raise ValueError(
            "Performance scores for each algorithm should have the same length and match k * num_repeats."
        )

    n = k * num_repeats

    d = [performance_A[i] - performance_B[i] for i in range(n)]
    d_bar = np.mean(d)
    s_d = np.std(d, ddof=1)

    rho = autocorrelation(d)
    effective_sample_size = n * (1 - rho) / (1 + rho)

    t_statistic = d_bar / (s_d / np.sqrt(effective_sample_size))
    degrees_of_freedom = effective_sample_size - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), degrees_of_freedom))

    reject_null_hypothesis = p_value < alpha

    return reject_null_hypothesis, t_statistic, p_value


import scipy.stats


def repeated_kfold_corrected_t_test(diff, r, k, n_train, n_test):
    diff_corrected = 1 / (k * r) * diff.sum()
    variance = 1 / (k * r - 1) * ((diff - diff_corrected) ** 2).sum()
    t = (
        1
        / (k * r)
        * diff_corrected
        / np.sqrt((1 / (k * r) + n_test / n_train) * variance)
    )

    p_value = scipy.stats.t.cdf(t, r * k - 1)
    return t

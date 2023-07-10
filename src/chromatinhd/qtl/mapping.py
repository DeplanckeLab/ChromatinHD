import numpy as np
import scipy.stats

def calculate_linreg_univariate_multiinput_multioutput(X, Y):
    # X = np.array([
    #     [0, 0, 0, 1, 1, 1, 2, 2],
    #     [0, 0, 0, 1, 1, 1, 2, 2],
    #     [0, 0, 0, 1, 1, 1, 2, 2],
    #     [0, 0, 0, 1, 1, 1, 2, 2],
    # ]).T[:, None, :]

    # Y = np.array([
    #     [1, 0, 3, 5, 8, 2, 7, 9],
    #     [1, 0, 3, 5, 8, 2, 7, 7]
    # ]).T[..., None]

    # manual_p_values = np.array([
    #     [scipy.stats.linregress(X[:, 0, j], Y[:, i, 0]).pvalue for i in range(Y.shape[1])] for j in range(X.shape[2])
    # ]).T

    # assert np.isclose(p, manual_p_values).all()

    slope = ((X - X.mean(0)) * (Y - Y.mean(0))).sum(0) / (((X - X.mean(0))**2).sum(0))
    intercept = (Y - (slope * X)).mean(0)

    params = np.stack([intercept, slope], 1)

    predictions = X * slope + intercept
    newX = np.pad(X, ((0, 0), (1, 0), (0, 0)), constant_values = 1)
    df = len(X) - 2 # 2 parameters
    MSE = (((Y-predictions)**2).sum(0))/df

    var_b = MSE[:, None]*np.stack([np.linalg.inv(np.dot(newX_.T,newX_)).diagonal() for newX_ in newX.transpose(2, 0, 1)], 1)
    sd_b = np.sqrt(var_b)
    ts_b = np.abs(params / sd_b)[:, 1]

    p = 2*(1-scipy.stats.t.cdf(ts_b,(len(newX)-len(newX[0]))))

    return slope, p




def calculate_pearson_univariate_multiinput_multioutput(X, Y):
    n = X.shape[0]

    cor = ((X - X.mean(0)) * (Y - Y.mean(0))).sum(0) / ((n - 1) * X.std(0) * Y.std(0))

    t = np.abs(cor * np.sqrt(n - 2) / np.sqrt(1 - cor**2))

    p = 2*(1-scipy.stats.t.cdf(t,n-2))

    return cor, p



def calculate_spearman_univariate_multiinput_multioutput(X, Y):
    # X = scipy.stats.rankdata(X, axis = 0)
    Y = scipy.stats.rankdata(Y, axis = 0)

    n = X.shape[0]

    cor = ((X - X.mean(0)) * (Y - Y.mean(0))).sum(0) / ((n - 1) * X.std(0) * Y.std(0))

    t = np.abs(cor * np.sqrt(n - 2) / np.sqrt(1 - cor**2))

    p = 2*(1-scipy.stats.t.cdf(t,n-2))

    return cor, p

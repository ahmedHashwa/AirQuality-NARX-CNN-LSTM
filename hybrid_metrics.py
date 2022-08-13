import numpy as np
from sklearn.metrics import mean_squared_error


def index_agreement(o, s):
    """
    index of agreement

    Willmott (1981, 1982)
    input:
        o: observed
        s: simulated
    output:
        ia: index of agreement
    """

    ia = 1 - (np.sum((o - s) ** 2)) / (np.sum(
        (np.abs(s - np.mean(o)) + np.abs(o - np.mean(o))) ** 2))
    return ia


def rmse(actual: np.ndarray, predicted: np.ndarray):
    """ Root Mean Squared Error """
    return mean_squared_error(y_true=actual, y_pred=predicted, squared=False)


def nrmse(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Root Mean Squared Error """
    return rmse(actual, predicted) / (actual.max() - actual.min())

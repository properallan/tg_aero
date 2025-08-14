import numpy as np


def nrse(x, y):
    # return (x-y)/(np.max(x)-np.min(x))
    return np.sqrt(np.square(np.subtract(x, y)))/(x.max() - y.min())*100


def nrmse(x, y):
    # return (x-y)/(np.max(x)-np.min(x))
    return np.sqrt(np.square(np.subtract(x, y)).mean())/(x.max() - y.min())*100
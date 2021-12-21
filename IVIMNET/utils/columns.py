import numpy as np


def column_creation(CV_net, CV_super, CV_dc):
    a = np.stack((CV_net, CV_super, CV_dc), axis=0)
    np_col = np.expand_dims(a, axis=0)
    return np_col


def pre_col(CV_net, CV_dc):
    a = np.stack((CV_net, CV_dc), axis=0)
    np_col = np.expand_dims(a, axis=0)
    return np_col
import numpy as np
from scipy.ndimage import minimum_filter

def find_maxima(data:np.ndarray)->np.ndarray:
    """
    Relative maxima finder

    :param data: 2D np array
    :return: (N, 2) maxima positions
    """
    data = -1*data
    maxima = (data == minimum_filter(data, 3, mode='constant', cval=0.0))
    res = np.array(np.where(1 == maxima)).T
    return res

def find_minima(data:np.ndarray)->np.ndarray:
    """
    Relative minima finder

    :param data: 2D np array
    :return: (N, 2) minima positions
    """
    minima = (data == minimum_filter(data, 3, mode='constant', cval=0.0))
    res = np.array(np.where(1 == minima)).T
    return res


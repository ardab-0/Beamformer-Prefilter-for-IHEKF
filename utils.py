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


def generate_spiral_path(a, theta_extent, alpha):
    theta = np.linspace(0, theta_extent)
    scaling = np.linspace(0.5, 1)

    x = a * np.cos(theta) * scaling
    y = a * np.sin(theta) * scaling
    z = a * theta * np.tan(alpha)
    return np.array([x, y, z]).reshape((3, -1))

def mse(x, x_hat):
    return np.mean(np.square(x - x_hat))



def mod_2pi(x):
    mod = np.mod(x, 2 * np.pi)
    mod[mod >= np.pi] -= 2 * np.pi
    return mod

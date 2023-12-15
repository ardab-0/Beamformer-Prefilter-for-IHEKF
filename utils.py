import numpy as np
from scipy.ndimage import minimum_filter
import math
from settings.config import Parameters as params
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
    theta = np.linspace(0, theta_extent, num=params.k)
    scaling = np.linspace(0.5, 1, num = params.k)

    x = a * np.cos(theta) * scaling
    y = a * np.sin(theta) * scaling
    z = a * theta * np.tan(alpha)
    return np.array([x, y, z]).reshape((3, -1))

def rmse(x, x_hat):
    return np.sqrt(np.mean(np.square(x - x_hat)))



def mod_2pi(x):
    mod = np.mod(x, 2 * np.pi)
    mod[mod >= np.pi] -= 2 * np.pi
    return mod


def cartesian_to_spherical(x, y, z):
    """

    :param x:
    :param y:
    :param z:
    :return: r, theta, phi
    """
    r = math.sqrt(x**2 + y**2 + z**2)
    phi = math.atan2(y, x)
    theta = math.acos(z / r)
    return r, theta, phi

import numpy as np
from scipy.ndimage import minimum_filter
import math
from settings.config import Parameters as params


def find_relative_maxima(data: np.ndarray, threshold: float = 0) -> np.ndarray:
    """
    Absolute maxima finder

    :param data: 2D np array
    :return: (N, 2) maxima positions
    """
    # data_max = np.max(data)
    # data /= data_max
    # data = 1 - data
    # maxima = (data == minimum_filter(data, 3, mode='constant', cval=0.0))
    data = np.pad(data, pad_width=1, mode="edge")
    d1 = data[1:-1, 1:-1] - data[1:-1, 2:]
    d2 = data[1:-1, 1:-1] - data[1:-1, 0:-2]
    d3 = data[1:-1, 1:-1] - data[0:-2, 0:-2]
    d4 = data[1:-1, 1:-1] - data[0:-2, 1:-1]
    d5 = data[1:-1, 1:-1] - data[0:-2, 2:]
    d6 = data[1:-1, 1:-1] - data[2:, 0:-2]
    d7 = data[1:-1, 1:-1] - data[2:, 1:-1]
    d8 = data[1:-1, 1:-1] - data[2:, 2:]
    m1 = d1 >= 0
    m2 = d2 >= 0
    m3 = d3 >= 0
    m4 = d4 >= 0
    m5 = d5 >= 0
    m6 = d6 >= 0
    m7 = d7 >= 0
    m8 = d8 >= 0
    m9 = data[1:-1, 1:-1] >= threshold

    maxima = m1 & m2 & m3 & m4 & m5 & m6 & m7 & m8 & m9

    # maxima = np.pad(maxima, pad_width=1, constant_values=False)

    res = np.array(np.where(1 == maxima)).T
    return res


def find_minima(data: np.ndarray) -> np.ndarray:
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
    scaling = np.linspace(0.5, 1, num=params.k)

    x = a * np.cos(theta) * scaling
    y = a * np.sin(theta) * scaling
    z = a * theta * np.tan(alpha) + 1
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
    r = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    phi = math.atan2(y, x)
    theta = math.acos(z / r)
    return r, theta, phi


def cartesian_to_spherical_np(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r += np.finfo(float).eps
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)

    return r, theta, phi


def spherical_to_cartesian_np(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z

def spherical_to_cartesian(r, theta, phi):
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)

    return x, y, z



def cone_filter(cartesian_points, target_theta, target_phi, cone_angle):
    """

    :param cartesian_points: 3xN
    :param target_theta: rad
    :param target_phi: rad
    :param cone_angle: rad
    :return:
    """
    dir = np.array([np.sin(target_theta) * np.cos(target_phi), np.sin(target_theta) * np.sin(target_phi),
                    np.cos(target_theta)])
    cartesian_points_normalized = cartesian_points / np.linalg.norm(cartesian_points, axis=0)
    projection = dir @ cartesian_points_normalized
    kept_idx = projection > np.cos(cone_angle)
    return kept_idx
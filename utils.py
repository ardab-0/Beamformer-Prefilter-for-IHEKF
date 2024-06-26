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
    a = np.linalg.norm(x - x_hat, axis=0)
    return np.sqrt(np.mean(np.square(a)))

def error_vector(x, x_hat):
    return np.linalg.norm(x - x_hat, axis=0)

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


def phase_error(x, x_hat):
    """

    :param x: N x Antenna element count
    :param x_hat: N x Antenna element count
    :return:
    """
    a = np.linalg.norm(x - x_hat, axis=0) / len(x)
    return np.mean(a)


def optimal_rotation_and_translation(A, B):
    """
        A: points (3xN)
        B: points (3xN)
        R: (3x3)
        t = (3x1)
        source: https://simonensemble.github.io/posts/2018-10-27-orthogonal-procrustes/
                https://nghiaho.com/?page_id=671
                https://en.wikipedia.org/wiki/Kabsch_algorithm
    """

    centroidA = np.mean(A, axis=1).reshape((-1, 1))
    centroidB = np.mean(B, axis=1).reshape((-1, 1))
    # print(centroidA)
    # print(centroidB)
    H = (A - centroidA) @ (B - centroidB).T

    U, S, Vh = np.linalg.svd(H)

    V = Vh.T
    # print(U.shape)
    # print(S.shape)
    # print(V.shape)

    R = V @ U.T

    if np.linalg.det(R) < 0:
        print("negative det")
        V[:, 2] *= -1
        R = V @ U.T

    t = centroidB - R @ centroidA

    return R, t


def plot_antennas_in_plane(antennas, array_nr: int, A_list: [] = []):
    middle = np.mean(antennas, axis=0)
    # define a plane
    mid_to_ant = middle[None, :] - antennas

    # ax = plt.figure().add_subplot(projection='3d')
    middle_direction = np.array([0.001, 0, 0])
    for v1 in mid_to_ant:
        for v2 in mid_to_ant:
            if not np.all(v1 == v2):
                a = np.cross(v1, v2)
                angle = np.arccos(a.dot(middle_direction) / (np.linalg.norm(a) * np.linalg.norm(middle_direction)))
                if angle < (np.pi / 2 + 1e-6):
                    middle_direction = middle_direction + a / np.linalg.norm(a)

    rot_vector = np.cross(middle_direction / np.linalg.norm(middle_direction), [0, 0, 1])
    rot_angle = np.arccos(middle_direction.dot([0, 0, 1]) / (np.linalg.norm(middle_direction)))
    mid_to_ant_rot = (KF.__rotation_matrix(rot_vector, rot_angle) @ mid_to_ant.T).T
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.scatter(mid_to_ant_rot[:,0],mid_to_ant_rot[:,1],mid_to_ant_rot[:,2])
    # KF_Wrapper.set_axes_equal(ax)
    ax = plt.figure().add_subplot()
    ax.axis("equal")
    ax.scatter(mid_to_ant_rot[:, 0], mid_to_ant_rot[:, 1], color="black")
    plt.title(f"Array nr: {array_nr}")
    for i in range(12): plt.text(mid_to_ant_rot[i, 0] + 0.001, mid_to_ant_rot[i, 1] + 0.001, str(i))
    if len(A_list) != 0:
        iterations = len(A_list)
        for i, (color, A) in enumerate(zip(["r", "y", "b", "g"], A_list[::-1])):
            for x in A:
                if np.isin(x, -2).any() or np.isin(x, 2).any():  # it is a diffdiff evaluation
                    [a1_index, a3_index] = [i for i, tmp in enumerate(x) if (tmp == 1 or tmp == -1)]
                    try:
                        a2_index = list(x).index(-2)
                    except:
                        a2_index = list(x).index(2)
                    ax.plot(mid_to_ant_rot[[a1_index, a2_index, a3_index], 0],
                            mid_to_ant_rot[[a1_index, a2_index, a3_index], 1],
                            color=color, label=f"Itaeration {iterations - i}: diffdiff")
                else:
                    a1_index = list(x).index(-1)
                    a2_index = list(x).index(1)
                    ax.plot(mid_to_ant_rot[[a1_index, a2_index], 0], mid_to_ant_rot[[a1_index, a2_index], 1],
                            color=color, label=f"Itaeration {iterations - i}")
    ax.legend()
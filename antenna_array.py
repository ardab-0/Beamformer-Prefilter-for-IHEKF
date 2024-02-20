import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from matplotlib import cm

import utils
from utils import cartesian_to_spherical


class AntennaArray:
    def __init__(self, rot: list, t: list, element_positions: np.ndarray):
        self._rot_list = rot
        self._R = self._compute_R(rot)
        self._t = self._compute_t(t)
        self._element_positions = element_positions
        theta, phi = self._compute_theta_phi()
        self.antenna_element = AntennaElement(mu_theta=theta, mu_phi=phi, sigma_phi=1, sigma_theta=1)

    def get_antenna_positions(self):
        ant_pos = self._R @ self._element_positions + self._t
        return ant_pos

    def _compute_R(self, rot: list):
        r = Rot.from_euler('xyz', rot, degrees=True)
        R = r.as_matrix()
        return R

    def _compute_t(self, t: list):
        return np.array([t]).T

    def get_R(self):
        return self._R

    def get_t(self):
        return self._t

    def _compute_theta_phi(self):
        initial_direction = np.array([[1, 0, 0]]).T

        new_direction = self._R @ initial_direction
        r, theta, phi = cartesian_to_spherical(new_direction[0], new_direction[1], new_direction[2])
        return theta, phi

    def get_antenna_element_beampattern(self, thetas, phis):
        return self.antenna_element.get_element_pattern(thetas, phis)


class RealDataAntennaArray:
    def __init__(self):
        self.antenna_positions = None

    def load_antenna_positions(self, antenna_pos):
        self.antenna_positions = antenna_pos

    def get_antenna_positions(self):
        return self.antenna_positions

    def get_t(self):
        return np.mean(self.antenna_positions, axis=1).reshape((3, 1))

class AntennaElement:
    def __init__(self, mu_theta=0, mu_phi=0, sigma_theta=0.3, sigma_phi=0.3):
        self.mu_theta = mu_theta
        self.mu_phi = mu_phi
        self.sigma_theta = sigma_theta
        self.sigma_phi = sigma_phi

    def get_element_pattern(self, thetas, phis):
        theta, phi = np.meshgrid(thetas, phis)
        shape = theta.shape
        dir = np.array([np.sin(self.mu_theta) * np.cos(self.mu_phi), np.sin(self.mu_theta) * np.sin(self.mu_phi), np.cos(self.mu_theta)])


        pattern = np.ones_like(theta)


        # t_i = np.logical_and(theta > theta_range[0], theta < theta_range[1])
        # p_i = np.logical_and(phi > phi_range[0], phi < phi_range[1])
        # id = np.logical_and(t_i, p_i)
        # pattern[id] = 1

        theta_1 = theta.ravel()
        phi_1 = phi.ravel()
        pattern_1 = pattern.ravel()

        cartesian = utils.spherical_to_cartesian_np(pattern_1, theta_1, phi_1)
        cartesian = np.vstack(cartesian)
        projection = dir @ cartesian
        pattern_1[projection < 0] = 0


        # rotated_spherical = utils.cartesian_to_spherical_np(rotated_cartesian[0], rotated_cartesian[1], rotated_cartesian[2])

        pattern_1 = pattern_1.reshape(shape)



        return pattern_1, theta, phi


# phis = np.linspace(-1 * np.pi, np.pi, 100)
# thetas = np.linspace(0, np.pi, 100)
#
# x = 0
# y = 0
# z = -1
# r, theta, phi = cartesian_to_spherical(x, y, z)
# del theta
# del phi
#
# ant = AntennaElement()
# pattern, theta, phi = ant.get_element_pattern(thetas, phis)
#
# x = np.abs(pattern) * np.sin(theta) * np.cos(phi)
# y = np.abs(pattern) * np.sin(theta) * np.sin(phi)
# z = np.abs(pattern) * np.cos(theta)
#
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(x, y, z, vmin=pattern.min() * 2, cmap=cm.Blues)
# plt.show()

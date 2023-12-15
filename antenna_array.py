import numpy as np
from scipy.spatial.transform import Rotation as Rot
import matplotlib.pyplot as plt
from matplotlib import cm

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

        new_direction = self._R @initial_direction
        r, theta, phi = cartesian_to_spherical(new_direction[0], new_direction[1], new_direction[2])
        return theta, phi

    def get_antenna_element_beampattern(self, thetas, phis):
        return self.antenna_element.get_element_pattern(thetas, phis)


class AntennaElement:
    def __init__(self, mu_theta=0, mu_phi=0, sigma_theta=0.3, sigma_phi=0.3):
        self.mu_theta = mu_theta
        self.mu_phi = mu_phi
        self.sigma_theta = sigma_theta
        self.sigma_phi = sigma_phi

    def get_element_pattern(self, theta, phi):
        theta, phi = np.meshgrid(theta, phi)

        # to get a periodic pattern
        #
        # phi period: 2 * pi
        pattern_theta = np.exp(-1 / 2 * ((theta - self.mu_theta) / self.sigma_theta) ** 2)
            #              + np.exp(
            # -1 / 2 * ((theta - self.mu_theta - np.pi) / self.sigma_theta) ** 2) + np.exp(
            # -1 / 2 * ((theta - self.mu_theta + np.pi) / self.sigma_theta) ** 2)
        pattern_phi = np.exp(-1 / 2 * ((phi - self.mu_phi) / self.sigma_phi) ** 2) + np.exp(
            -1 / 2 * ((phi - self.mu_phi - 2 * np.pi) / self.sigma_phi) ** 2) + np.exp(
            -1 / 2 * ((phi - self.mu_phi + 2 * np.pi) / self.sigma_phi) ** 2)
        pattern = pattern_theta * pattern_phi
        max = np.max(pattern)
        pattern /= max
        return pattern, theta, phi



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

import numpy as np
from scipy.spatial.transform import Rotation as Rot

class AntennaArray:
    def __init__(self, rot: list, t: list, element_positions: np.ndarray):
        self._rot_list = rot
        self._R = self._compute_R(rot)
        self._t = self._compute_t(t)
        self._element_positions = element_positions
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


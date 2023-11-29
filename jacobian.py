from sympy import MatrixSymbol, Matrix
import sympy as s
import numpy as np


class Jacobian_h():
    def __init__(self, N, I, w0, c):
        self.w0 = w0
        self.c = c
        self.N = N
        self.I = I
        self.H = None

    def compute_jacobian(self):

        self.x, self.y, self.z, self.vx, self.vy, self.vz = s.symbols('x, y, z, vx, vy, vz')

        pB = s.Matrix([self.x, self.y, self.z])
        self.A = MatrixSymbol('A', self.N, self.I)
        phi_mix = MatrixSymbol("phi_mix", self.I, 1)

        self.pMs = []

        for i in range(self.I):
            pM = MatrixSymbol(f"pM^{i}", 3, 1)
            self.pMs.append(pM)

            tau_row = s.sqrt((pB - pM).T * (pB - pM)) / self.c
            tau_row = s.Matrix(tau_row)
            if i == 0:
                tau = tau_row
            else:
                tau = tau.row_insert(i, tau_row)

        phi = - self.w0 * tau + phi_mix
        h = self.A * phi

        h = h.as_explicit()
        self.H = h.jacobian(Matrix([self.x, self.y, self.z, self.vx, self.vy, self.vz]))

    def evaluate_jacobian(self, A_np, px, py, pz, ant_pos):

        subs = {self.A: s.Matrix(A_np),
                self.x: px,
                self.y: py,
                self.z: pz}

        for i, pM in enumerate(self.pMs):
            subs[pM] = s.Matrix(ant_pos[:, i])

        F = np.array(self.H.evalf(subs=subs)).astype(float)
        return F



def jacobian_numpy(A_np, px, py, pz, ant_pos, c, w0):
    p = np.array([[px, py, pz]]).T
    u = p - ant_pos
    u_norm = np.linalg.norm(u, axis=0)
    k = u / u_norm

    dxdydz = -w0 / c * A_np @ k.T

    jacobian = np.zeros((dxdydz.shape[0], 6)) # append dvx dvy dvz
    jacobian[:, :3] = dxdydz
    return jacobian

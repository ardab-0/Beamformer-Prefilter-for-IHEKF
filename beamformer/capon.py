import numpy as np
from settings.config import Parameters as params


class CaponBeamformer:
    def __init__(self, type):
        """
        3D beamformer
        :param type: "cpu_np" or "gpu" or "cpu"
        """
        self.type = type

    def compute_beampattern(self, x, N_theta, N_phi, fs, r, phi=0, theta=0):
        """

        :param x: input signals, shape:(N_array, N)
        :param N_theta: theta sample number
        :param fs: sampling freq
        :param r: antenna positions, shape: (3,N)
        :param N_phi: phi sample number
        :return: (  results: signal power at (theta, phi), shape: (N_theta, N_phi),
                    output_signals: signal at (theta, phi), shape: (N_theta, N_phi, N),
                    thetas: shape: (N_theta,),
                    phis: shape: (N_phi,)
                )
        """
        if N_theta == 1 and N_phi == 1:
            thetas = theta
            phis = phi
        elif N_theta == 1:
            phis = np.linspace(-1 * np.pi, np.pi, N_phi)
            thetas = theta
        elif N_phi == 1:
            thetas = np.linspace(-1 * np.pi, np.pi, N_theta)
            phis = phi
        else:
            phis = np.linspace(-1 * np.pi, np.pi, N_phi)
            thetas = np.linspace(0, np.pi, N_theta)

        if self.type == "cpu_np":
            return self.__compute_beampatern_cpu_np(x, thetas, phis, fs, r)
        elif self.type == "gpu":
            return self.__compute_beampatern_gpu(x, thetas, phis, fs, r)
        elif self.type == "cpu":
            return self.__compute_beampatern_cpu(x, thetas, phis, fs, r)
        else:
            raise TypeError("Wrong compute type.")


    def __compute_beampatern_cpu_np(self, x, N_theta, N_phi, fs, r):
        raise NotImplementedError
    def __compute_beampatern_gpu(self, x, N_theta, N_phi, fs, r):
        raise NotImplementedError



    def __compute_beampatern_cpu(self, x, thetas, phis, fs, r):
        N_array, N = x.shape

        output_signals = np.zeros((len(thetas), len(phis), N), dtype=np.complex64)
        results = np.zeros((len(thetas), len(phis)))
        x = np.asmatrix(x)
        # Calc covariance matrix
        R = x @ x.H  # gives a Nr x Nr covariance matrix of the samples

        Rinv = np.linalg.pinv(R)  # pseudo-inverse tends to work better than a true inverse
        for k, theta_i in enumerate(thetas):
            for l, phi in enumerate(phis):
                u_sweep = np.array([np.sin(theta_i) * np.cos(phi), np.sin(theta_i) * np.sin(phi), np.cos(theta_i)]).reshape(
                    (-1, 1))
                a = r.T @ u_sweep
                a = np.exp(1j * 2 * np.pi * params.f * a / params.c)
                a = np.asmatrix(a)
                c = 1 / (a.H @ Rinv @ a)
                w = Rinv @ a * c
                c = c[0, 0].real
                results[k, l] = np.square(c)
                out = w.H @ x
                output_signals[k, l, :] = out

        results /= np.max(results)  # normalize
        output_signals = np.transpose(output_signals, (1, 0, 2))
        return results.T, output_signals, thetas, phis


    def spherical_to_cartesian(self, results, thetas, phis):
        """

        :param results: signal power at (theta, phi), shape: (N_theta, N_phi)
        :param thetas: shape: (N_theta, )
        :param phis: shape: (N_phi, )
        :return: cartesian pointcloud data shape: (3, N_theta x N_phi)
        """
        theta_mesh, phi_mesh = np.meshgrid(thetas, phis)
        r = np.array(results).reshape((-1, 1))
        theta = theta_mesh.reshape((-1, 1))
        phi = phi_mesh.reshape((-1, 1))


        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.array([x, y, z]).reshape((3, -1))

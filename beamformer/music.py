import numpy as np
from settings.config import Parameters as params


class MusicBeamformer:
    def __init__(self, type):
        """
        3D beamformer
        :param type: "cpu_np" or "gpu" or "cpu"
        """
        self.type = type

    def compute_beampattern(self, x, N_theta, N_phi, fs, r):
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
        if self.type == "cpu_np":
            return self.__compute_beampatern_cpu_np(x, N_theta, N_phi, fs, r)
        elif self.type == "gpu":
            return self.__compute_beampatern_gpu(x, N_theta, N_phi, fs, r)
        elif self.type == "cpu":
            return self.__compute_beampatern_cpu(x, N_theta, N_phi, fs, r)
        else:
            raise TypeError("Wrong compute type.")


    def __compute_beampatern_cpu_np(self, x, N_theta, N_phi, fs, r):
        raise NotImplementedError
    def __compute_beampatern_gpu(self, x, N_theta, N_phi, fs, r):
        raise NotImplementedError



    def __compute_beampatern_cpu(self, x, N_theta, N_phi, fs, r):
        num_expected_signals = 3  # Try changing this!
        N_array = len(r[0, :])

        x = np.asmatrix(x)
        # part that doesn't change with theta_i
        R = x @ x.H  # Calc covariance matrix, it's Nr x Nr
        w, v = np.linalg.eig(
            R)  # eigenvalue decomposition, v[:,i] is the eigenvector corresponding to the eigenvalue w[i]
        eig_val_order = np.argsort(np.abs(w))  # find order of magnitude of eigenvalues
        v = v[:, eig_val_order]  # sort eigenvectors using this order
        # We make a new eigenvector matrix representing the "noise subspace", it's just the rest of the eigenvalues
        V = np.asmatrix(np.zeros((N_array, N_array - num_expected_signals), dtype=np.complex64))
        for i in range(N_array - num_expected_signals):
            V[:, i] = v[:, i]

        thetas = np.linspace(0, np.pi, N_theta)
        phis = np.linspace(-1 * np.pi, np.pi, N_phi)

        results = np.zeros((N_theta, N_phi))

        for k, theta_i in enumerate(thetas):
            for l, phi in enumerate(phis):
                u_sweep = np.array(
                    [np.sin(theta_i) * np.cos(phi), np.sin(theta_i) * np.sin(phi), np.cos(theta_i)]).reshape(
                    (-1, 1))
                a = r.T @ u_sweep
                a = np.exp(1j * 2 * np.pi * params.f * a / params.c)
                a = np.asmatrix(a)
                metric = 1 / (a.H @ V @ V.H @ a)  # The main MUSIC equation
                metric = np.abs(metric[0, 0])  # take magnitude

                results[k, l] = np.square(metric)

        results /= np.max(results)  # normalize
        results = np.sqrt(results)  # power to amplitude conversion
        return results.T, results, thetas, phis


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

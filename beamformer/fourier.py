import numpy as np
from settings.config import Parameters as params
import torch

class FourierBeamformer:
    def __init__(self, type):
        """
        3D beamformer
        :param type: "cpu_np" or "gpu" or "cpu"
        """
        self.type = type

    def compute_beampattern(self, x, N_theta, N_phi, fs, r, phi=0, theta=90):
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
            thetas = np.linspace(-1*np.pi, np.pi, N_theta)
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
            raise TypeError("Wrong beamfomer type.")


    def __compute_beampatern_cpu_np(self, x, thetas, phis, fs, r):
        N_array, N = x.shape

        x_fft = np.fft.fft(x, axis=1)

        f = np.fft.fftfreq(N, 1 / fs).reshape((1, -1))
        # results = np.zeros((N_theta, N_phi))
        #
        # output_signals = np.zeros((N_theta, N_phi, N), dtype=np.complex64)
        theta_sweep, phi_sweep = np.meshgrid(thetas, phis)
        u_sweep = np.array(
            [np.sin(theta_sweep) * np.cos(phi_sweep), np.sin(theta_sweep) * np.sin(phi_sweep),
             np.cos(theta_sweep)])

        v = np.tensordot(r.T, u_sweep, axes=1)
        v = np.expand_dims(v, axis=1)
        f = np.expand_dims(f, axis=(2, 3))
        H = np.exp(-1j * 2 * np.pi * f * v / params.c)
        x_fft = np.expand_dims(x_fft, axis=(2, 3))
        out = np.sum(x_fft * H, axis=0)
        out /= N_array
        out = np.fft.ifft(out, axis=0)
        output_signals = np.transpose(out, (1, 2, 0))
        results = np.mean(np.abs(out) ** 2, axis=0)
        results /= np.max(results)
        return results, output_signals, thetas, phis

    def __compute_beampatern_gpu(self, x, thetas, phis, fs, r):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        N_array, N = x.shape

        x = torch.from_numpy(x)
        x_fft = torch.fft.fft(x, axis=1)



        f = torch.fft.fftfreq(N, 1 / fs).reshape((1, -1))
        # results = np.zeros((N_theta, N_phi))
        #
        # output_signals = np.zeros((N_theta, N_phi, N), dtype=np.complex64)
        theta_sweep, phi_sweep = np.meshgrid(thetas, phis)

        u_sweep = np.array(
            [np.sin(theta_sweep) * np.cos(phi_sweep), np.sin(theta_sweep) * np.sin(phi_sweep),
             np.cos(theta_sweep)])
        u_sweep = torch.from_numpy(u_sweep)


        r = torch.from_numpy(r)
        r.type(torch.float32)


        v = torch.tensordot(r.T, u_sweep, dims=1)
        v = torch.unsqueeze(v, dim=1)

        f = torch.unsqueeze(f, dim=2)
        f = torch.unsqueeze(f, dim=3)
        u = f * v
        u = u.to(device)

        H = torch.exp(-1j * 2 * np.pi * u / params.c)
        x_fft = torch.unsqueeze(x_fft, dim=2)
        x_fft = torch.unsqueeze(x_fft, dim=3)
        x_fft = x_fft.to(device)
        out = torch.sum(x_fft * H, dim=0)
        out /= N_array
        out = torch.fft.ifft(out, axis=0)
        output_signals = torch.permute(out, (1, 2, 0))
        results = torch.mean(torch.abs(out) ** 2, dim=0)
        results /= torch.max(results)
        return results.cpu().numpy(), output_signals.cpu().numpy(), thetas, phis



    def __compute_beampatern_cpu(self, x, thetas, phis, fs, r):
        N_array, N = x.shape
        x_fft = np.zeros((N_array, N), dtype=complex)
        for i in range(N_array):
            x_fft[i, :] = np.fft.fft(x[i, :])


        f = np.fft.fftfreq(N, 1 / fs)
        results = np.zeros((len(thetas), len(phis)))

        output_signals = np.zeros((len(thetas), len(phis), N), dtype=np.complex64)
        for k, theta_sweep in enumerate(thetas):
            for l, phi_sweep in enumerate(phis):
                u_sweep = np.array(
                    [np.sin(theta_sweep) * np.cos(phi_sweep), np.sin(theta_sweep) * np.sin(phi_sweep), np.cos(theta_sweep)])

                out = 0
                for i in range(N_array):
                    H = np.exp(-1j * 2 * np.pi * f * np.dot(u_sweep, r[:, i]) / params.c)
                    out += x_fft[i, :] * H
                out /= N_array
                out = np.fft.ifft(out)
                output_signals[k, l, :] = out
                results[k, l] = np.mean(np.abs(out) ** 2)
        results /= np.max(results)
        return results, output_signals, thetas, phis


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

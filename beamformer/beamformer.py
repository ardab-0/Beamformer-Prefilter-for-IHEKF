import numpy as np
from config import Parameters as params


def compute_beampatern(x, N_theta, N_phi, fs, r):
    """

    :param x: input signals (N_array x N)
    :param N_theta: theta sample number
    :param fs: sampling freq
    :param r: antenna positions (3xN np array)
    :param N_phi: phi sample number
    :return:
    """
    N_array, N = x.shape
    x_fft = np.zeros((N_array, N), dtype=complex)
    for i in range(N_array):
        x_fft[i, :] = np.fft.fft(x[i, :])

    thetas = np.linspace(-1 * np.pi, np.pi, N_theta)
    phis = np.linspace(-1 * np.pi, np.pi, N_phi)
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
    output_signals = out.reshape((N_theta, N_phi, -1))
    results = np.mean(np.abs(out) ** 2, axis=0)
    # for k, theta_sweep in enumerate(thetas):
    #     for l, phi_sweep in enumerate(phis):
    #         u_sweep = np.array(
    #             [np.sin(theta_sweep) * np.cos(phi_sweep), np.sin(theta_sweep) * np.sin(phi_sweep), np.cos(theta_sweep)]).reshape((3, 1))
    #
    #
    #         v = r.T @ u_sweep
    #         H = np.exp(-1j * 2 * np.pi * f * v / params.c)
    #         out = np.sum(x_fft * H, axis=0)
    #         out /= N_array
    #         out = np.fft.ifft(out)
    #         output_signals[k, l, :] = out
    #         results[k, l] = np.mean(np.abs(out) ** 2)
    return results, output_signals, thetas, phis


def compute_beampatern_orig(x, N_theta, N_phi, fs, r):
    """

    :param x: input signals (N_array x N)
    :param N_theta: theta sample number
    :param fs: sampling freq
    :param r: antenna positions (3xN np array)
    :param N_phi: phi sample number
    :return:
    """
    N_array, N = x.shape
    x_fft = np.zeros((N_array, N), dtype=complex)
    for i in range(N_array):
        x_fft[i, :] = np.fft.fft(x[i, :])

    thetas = np.linspace(-1 * np.pi, np.pi, N_theta)
    phis = np.linspace(-1 * np.pi, np.pi, N_phi)
    f = np.fft.fftfreq(N, 1 / fs)
    results = np.zeros((N_theta, N_phi))

    output_signals = np.zeros((N_theta, N_phi, N), dtype=np.complex64)
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
    return results, output_signals, thetas, phis


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z]).reshape((3, -1))

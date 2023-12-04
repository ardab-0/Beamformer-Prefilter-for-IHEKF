import numpy as np
from config import Parameters as params

def compute_beampatern(x, N_theta, fs, r, phi=0):
    N_array, N = x.shape
    x_fft = np.zeros((N_array, N), dtype=complex)
    for i in range(N_array):
        x_fft[i, :] = np.fft.fft(x[i, :])

    thetas = np.linspace(-1 * np.pi, np.pi, N_theta)
    f = np.fft.fftfreq(N, 1 / fs)
    results = []

    output_signals = np.zeros((N_theta, N), dtype=np.complex64)
    for k, theta_sweep in enumerate(thetas):
        u_sweep = np.array([np.sin(theta_sweep) * np.cos(phi), np.sin(theta_sweep) * np.sin(phi), np.cos(theta_sweep)])

        out = 0
        for i in range(N_array):
            H = np.exp(-1j * 2 * np.pi * f * np.dot(u_sweep, r[i]) / params.c)
            out += x_fft[i, :] * H
        out /= N_array
        out = np.fft.ifft(out)
        output_signals[k, :] = out
        results.append(np.mean(np.abs(out) ** 2))
    return results, output_signals, thetas
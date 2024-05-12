import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

# define paramters
phi = 0.0

theta1 = 60.0 / 180 * np.pi
theta2 = 120.0 / 180 * np.pi
theta3 = 75.0 / 180 * np.pi
N_array = 8

f = 24e9
c = 2.998e8

d = c / f / 2

N_theta = 1000
# define time vector
N = 1500
fs = 100 * f
t = np.arange(N) / fs

# define vectors u, r1, r2

r = np.zeros((N_array, 3))
x = np.zeros((N_array, N), dtype=complex)

for i in range(N_array):
    r[i, :] = np.array([0, 0, i * d])

# r[15, 2] = 0.25


print(r)



def array_signal_multiple_source(t, f_list, u_list, a_list, r, n_std=0.1):
    x = 0
    for f, u, a in zip(f_list, u_list, a_list):
        x += a * np.exp(1j * 2 * np.pi * f * (t + np.dot(u, r) / c))

    n = np.random.randn(N) + 1j * np.random.randn(N)
    return x + n_std * n


def array_signal(t, f1, f2, u1, u2, r):
    x1 = np.exp(1j * 2 * np.pi * f1 * (t + np.dot(u1, r) / c))
    x2 = 0.5 * np.exp(1j * 2 * np.pi * f2 * (t + np.dot(u2, r) / c))
    n = np.random.randn(N) + 1j * np.random.randn(N)

    return (x1 + x2 + 0.1 * n)


def first_component(t, f1, u1, r):
    x1 = np.exp(1j * 2 * np.pi * f1 * (t + np.dot(u1, r) / c))
    return x1


def compute_phase_shift(x, f, u, r):
    return x * np.exp(1j * 2 * np.pi * f * np.dot(u, r) / c)


def compute_beampatern(x):
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
            H = np.exp(-1j * 2 * np.pi * f * np.dot(u_sweep, r[i]) / c)
            out += x_fft[i, :] * H
        out /= N_array
        out = np.fft.ifft(out)
        output_signals[k, :] = out
        results.append(np.mean(np.abs(out) ** 2))
    return results, output_signals, thetas


def __compute_beampatern_cpu_np(x, phi=0):
    N_array, N = x.shape

    x_fft = np.fft.fft(x, axis=1)

    thetas = np.linspace(-1 * np.pi, np.pi, N_theta)
    f = np.fft.fftfreq(N, 1 / fs).reshape((1, -1))
    # results = np.zeros((N_theta, N_phi))
    #
    # output_signals = np.zeros((N_theta, N_phi, N), dtype=np.complex64)
    u_sweep = np.array(
        [np.sin(thetas) * np.cos(phi), np.sin(thetas) * np.sin(phi),
         np.cos(thetas)])

    v = np.tensordot(r, u_sweep, axes=1)
    v = np.expand_dims(v, axis=1)
    f = np.expand_dims(f, axis=(2))
    H = np.exp(-1j * 2 * np.pi * f * v / c)
    x_fft = np.expand_dims(x_fft, axis=(2))
    out = np.sum(x_fft * H, axis=0)
    out /= N_array
    out = np.fft.ifft(out, axis=0)
    output_signals = out.T
    results = np.sqrt( np.mean(np.abs(out) ** 2, axis=0) )

    return results, output_signals, thetas

# compute signal at antenna elements
u_list = [np.array([np.sin(theta1) * np.cos(phi), np.sin(theta1) * np.sin(phi), np.cos(theta1)]),
          # np.array([np.sin(theta2) * np.cos(phi), np.sin(theta2) * np.sin(phi), np.cos(theta2)]),
          #   np.array([np.sin(theta3) * np.cos(phi), np.sin(theta3) * np.sin(phi), np.cos(theta3)])
          ]

f_list = [f, ]
a_list = [1, ]
for i in range(N_array):
    x[i, :] = array_signal_multiple_source(t, f_list, u_list, a_list, r[i])

results1, output_signals1, thetas1 = __compute_beampatern_cpu_np(x)


u_list = [np.array([np.sin(theta2) * np.cos(phi), np.sin(theta2) * np.sin(phi), np.cos(theta2)]),
          # np.array([np.sin(theta2) * np.cos(phi), np.sin(theta2) * np.sin(phi), np.cos(theta2)]),
          #   np.array([np.sin(theta3) * np.cos(phi), np.sin(theta3) * np.sin(phi), np.cos(theta3)])
          ]

f_list = [f, ]
a_list = [0.2, ]
for i in range(N_array):
    x[i, :] = array_signal_multiple_source(t, f_list, u_list, a_list, r[i])

results2, output_signals2, thetas2 = __compute_beampatern_cpu_np(x)



plt.figure()

plt.plot(thetas1 * 180 / np.pi, results1, label="60 deg")
plt.plot(thetas1 * 180 / np.pi, results2, label="120 deg")

plt.xlabel("Theta (Degrees)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(thetas1, results1, label="60 deg")
ax.plot(thetas1, results2, label="120 deg")# MAKE SURE TO USE RADIAN FOR POLAR
ax.set_theta_zero_location('N')  # make 0 degrees point up
ax.set_theta_direction(-1)  # increase clockwise
ax.set_rlabel_position(22.5)  # Move grid labels away from other labels
plt.legend()
#
# removed_component_idx = 1
#
# theta_to_remove = thetas[sorted_maxima[removed_component_idx]]
# u = np.array([np.sin(theta_to_remove) * np.cos(phi), np.sin(theta_to_remove) * np.sin(phi), np.cos(theta_to_remove)])
# signal_to_remove = output_signals[sorted_maxima[removed_component_idx]]
#
# signal_to_remove_at_antenna = np.zeros((N_array, N), dtype=complex)
# for i in range(N_array):
#     signal_to_remove_at_antenna[i, :] = compute_phase_shift(signal_to_remove, f, u, r[i])
#
# filtered_x = x - signal_to_remove_at_antenna
#
# results, output_signals, thetas = __compute_beampatern_cpu_np(filtered_x)
#
# plt.figure()
# plt.plot(thetas * 180 / np.pi, results)  # lets plot angle in degrees
# plt.xlabel("Theta [Degrees]")
# plt.ylabel("DOA Metric")
# plt.grid()
#
# fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
# ax.plot(thetas, results)  # MAKE SURE TO USE RADIAN FOR POLAR
# ax.set_theta_zero_location('N')  # make 0 degrees point up
# ax.set_theta_direction(-1)  # increase clockwise
# ax.set_rlabel_position(22.5)  # Move grid labels away from other labels

plt.show()

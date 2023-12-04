import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema

# define paramters
phi = 0.0

theta1 = 60.0 / 180 * np.pi
theta2 = 30.0 / 180 * np.pi
theta3 = 75.0 / 180 * np.pi
N_array = 16

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

u_list = [np.array([np.sin(theta1) * np.cos(phi), np.sin(theta1) * np.sin(phi), np.cos(theta1)]),
          np.array([np.sin(theta2) * np.cos(phi), np.sin(theta2) * np.sin(phi), np.cos(theta2)]),
            np.array([np.sin(theta3) * np.cos(phi), np.sin(theta3) * np.sin(phi), np.cos(theta3)])
          ]

f_list = [f, f, f]
a_list = [1, 0.5, 0.3]

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


# compute signal at antenna elements
for i in range(N_array):
    x[i, :] = array_signal_multiple_source(t, f_list, u_list, a_list, r[i])

results, output_signals, thetas = compute_beampatern(x)

real_out = output_signals.real
results = np.array(results)
maxima = argrelextrema(results, np.greater)[0]
max_val = results[maxima]

max_val = max_val[:int(len(max_val) / 2)]
arg_max_val = max_val.argsort()[::-1]

sorted_maxima = maxima[arg_max_val]
sorted_max_val = max_val[arg_max_val]
print(sorted_maxima)
print(sorted_max_val)

m = output_signals[sorted_maxima[0]]

plt.plot(output_signals[sorted_maxima[0]].real, label="0")
plt.plot(output_signals[sorted_maxima[1]].real, label="1")
plt.plot(x[0, :].real, label="ant0")
plt.plot(x[1, :].real, label="ant1")
plt.plot(x[2, :].real, label="ant2")
plt.plot(x[3, :].real, label="ant3")

plt.legend()
plt.figure()

plt.plot(thetas * 180 / np.pi, results)  # lets plot angle in degrees
plt.xlabel("Theta [Degrees]")
plt.ylabel("DOA Metric")
plt.grid()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(thetas, results)  # MAKE SURE TO USE RADIAN FOR POLAR
ax.set_theta_zero_location('N')  # make 0 degrees point up
ax.set_theta_direction(-1)  # increase clockwise
ax.set_rlabel_position(22.5)  # Move grid labels away from other labels

removed_component_idx = 1

theta_to_remove = thetas[sorted_maxima[removed_component_idx]]
u = np.array([np.sin(theta_to_remove) * np.cos(phi), np.sin(theta_to_remove) * np.sin(phi), np.cos(theta_to_remove)])
signal_to_remove = output_signals[sorted_maxima[removed_component_idx]]

signal_to_remove_at_antenna = np.zeros((N_array, N), dtype=complex)
for i in range(N_array):
    signal_to_remove_at_antenna[i, :] = compute_phase_shift(signal_to_remove, f, u, r[i])

filtered_x = x - signal_to_remove_at_antenna

results, output_signals, thetas = compute_beampatern(filtered_x)

plt.figure()
plt.plot(thetas * 180 / np.pi, results)  # lets plot angle in degrees
plt.xlabel("Theta [Degrees]")
plt.ylabel("DOA Metric")
plt.grid()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(thetas, results)  # MAKE SURE TO USE RADIAN FOR POLAR
ax.set_theta_zero_location('N')  # make 0 degrees point up
ax.set_theta_direction(-1)  # increase clockwise
ax.set_rlabel_position(22.5)  # Move grid labels away from other labels

plt.show()

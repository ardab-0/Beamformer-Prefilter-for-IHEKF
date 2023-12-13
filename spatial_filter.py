import utils
import numpy as np
from config import Parameters as params
from scipy.signal import argrelextrema

def remove_components_2D(x, r, results, phis, thetas, output_signals):
    filtered_x = x.copy()
    N_array = len(r[0])
    maxima = utils.find_maxima(results)
    max_val = results[maxima[:, 0], maxima[:, 1]]
    arg_max_val = max_val.argsort()[::-1]
    sorted_maxima = maxima[arg_max_val]

    for k in range(1, 2):
        theta_to_remove = thetas[sorted_maxima[k][1]]
        phi_to_remove = phis[sorted_maxima[k][0]]
        print(f"theta and phi to remove (rad): {theta_to_remove}, {phi_to_remove}")

        u = np.array(
            [np.sin(theta_to_remove) * np.cos(phi_to_remove), np.sin(theta_to_remove) * np.sin(phi_to_remove),
             np.cos(theta_to_remove)])
        signal_to_remove = output_signals[sorted_maxima[k][0], sorted_maxima[k][1]]

        signal_to_remove_at_antenna = np.zeros((N_array, params.N), dtype=complex)
        for i in range(N_array):
            signal_to_remove_at_antenna[i, :] = compute_phase_shift(signal_to_remove, params.f, u, r[:, i])

        filtered_x -= signal_to_remove_at_antenna

    print("\n\n")
    return filtered_x


def remove_components_1D_theta(x, r, results, phi, thetas, output_signals):
    filtered_x = x.copy()
    N_array = len(r[0])
    thetas = np.squeeze(thetas)
    results = np.squeeze(results)
    output_signals = np.squeeze(output_signals)

    thetas_to_include = thetas > 0  ############# might need to change
    thetas = thetas[thetas_to_include]
    results = results[thetas_to_include]
    output_signals = output_signals[thetas_to_include]

    maxima = argrelextrema(results, np.greater)[0]
    max_val = results[maxima]

    # max_val = max_val[:int(len(max_val) / 2)]
    arg_max_val = max_val.argsort()[::-1]

    sorted_maxima = maxima[arg_max_val]

    for k in range(1, 2):
        theta_to_remove = thetas[sorted_maxima[k]]
        print("theta to remove: ", theta_to_remove)
        u = np.array(
            [np.sin(theta_to_remove) * np.cos(phi), np.sin(theta_to_remove) * np.sin(phi), np.cos(theta_to_remove)])
        signal_to_remove = output_signals[sorted_maxima[k]]

        signal_to_remove_at_antenna = np.zeros((N_array, params.N), dtype=complex)
        for i in range(N_array):
            signal_to_remove_at_antenna[i, :] = compute_phase_shift(signal_to_remove, params.f, u, r[:, i])

        filtered_x -= signal_to_remove_at_antenna

    print("\n\n")
    return filtered_x
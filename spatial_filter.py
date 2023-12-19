from itertools import combinations

import utils
import numpy as np
from measurement_simulation import compute_phase_shift
from settings.config import Parameters as params
from scipy.signal import argrelextrema


def remove_close_peaks(theta_peak, phi_peak, eps = 0.1):
    """
    needs further testing

    :param theta_peak:
    :param phi_peak:
    :param eps:
    :return:
    """
    x, y, z = utils.spherical_to_cartesian_np(1, theta_peak, phi_peak)
    N = len(x)
    ids = np.arange(N)
    ids_pairs = np.array(list(combinations(ids, 2)))
    idx_to_keep = ids
    if len(ids_pairs) > 0:
        x_pairs = np.stack([x[ids_pairs[:, 0]], x[ids_pairs[:, 1]]]).T
        dx_pairs = np.diff(x_pairs, axis=1).ravel()
        y_pairs = np.stack([y[ids_pairs[:, 0]], y[ids_pairs[:, 1]]]).T
        dy_pairs = np.diff(y_pairs, axis=1).ravel()
        z_pairs = np.stack([z[ids_pairs[:, 0]], z[ids_pairs[:, 1]]]).T
        dz_pairs = np.diff(z_pairs, axis=1).ravel()

        d_pairs = np.sqrt(dx_pairs ** 2 + dy_pairs ** 2 + dz_pairs**2)

        close_pairs = ids_pairs[d_pairs < eps]
        idx_to_remove = set(close_pairs[:, 1])
        idx_set = set(ids)
        if len(idx_to_remove)>0:
            idx_set = idx_set - idx_to_remove
        idx_to_keep = list(idx_set)

    return theta_peak[idx_to_keep], phi_peak[idx_to_keep]

def remove_components_2D(x, r, results, phis, thetas, output_signals):
    filtered_x = x.copy()
    N_array = len(r[0])
    maxima = utils.find_relative_maxima(results, threshold=0.1)
    max_val = results[maxima[:, 0], maxima[:, 1]]
    arg_max_val = max_val.argsort()[::-1]
    sorted_maxima = maxima[arg_max_val]

    peak_thetas = thetas[sorted_maxima[:, 1]]
    peak_phis = phis[sorted_maxima[:, 0]]
    filtered_peak_thetas, filtered_peak_phis = remove_close_peaks(peak_thetas, peak_phis)

    for k in range(1, min(2, len(filtered_peak_thetas))):
        theta_to_remove = filtered_peak_thetas[k]
        phi_to_remove = filtered_peak_phis[k]
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
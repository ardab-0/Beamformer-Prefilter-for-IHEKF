from itertools import combinations
import utils
import numpy as np

from measurement_simulation import compute_phase_shift
from settings.config import Parameters as params
from scipy.signal import argrelextrema


def remove_close_peaks(theta_peak, phi_peak, sorted_maxima, eps=0.1):
    """
    removes peaks that are closer than eps
    needs further testing

    :param theta_peak: thetas of peaks
    :param phi_peak: phis of peaks
    :param eps: threshold to consider peaks close
    :return: filtered peak theta, phi and maximas
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

        d_pairs = np.sqrt(dx_pairs ** 2 + dy_pairs ** 2 + dz_pairs ** 2)

        close_pairs = ids_pairs[d_pairs < eps]
        idx_to_remove = set(close_pairs[:, 1])
        idx_set = set(ids)
        if len(idx_to_remove) > 0:
            idx_set = idx_set - idx_to_remove
        idx_to_keep = list(idx_set)

    return theta_peak[idx_to_keep], phi_peak[idx_to_keep], sorted_maxima[idx_to_keep]


def remove_components_2D(x, r, results, phis, thetas, output_signals, peak_threshold=params.peak_threshold):
    filtered_x = x.copy()
    N_array = len(r[0])

    # # due to periodicity of phi
    # results = np.vstack((results, results, results))
    # thetas = np.hstack((thetas, thetas, thetas))
    # phis = np.hstack((phis, phis, phis))
    # output_signals = np.vstack((output_signals, output_signals, output_signals))

    maxima = utils.find_relative_maxima(results, threshold=peak_threshold)
    max_val = results[maxima[:, 0], maxima[:, 1]]
    arg_max_val = max_val.argsort()[::-1]
    sorted_maxima = maxima[arg_max_val]

    peak_thetas = thetas[sorted_maxima[:, 1]]
    peak_phis = phis[sorted_maxima[:, 0]]
    filtered_peak_thetas, filtered_peak_phis, sorted_maxima = remove_close_peaks(peak_thetas, peak_phis, sorted_maxima,
                                                                                 eps=0.2)

    for k in range(1, min(params.num_peaks_to_remove + 1, len(filtered_peak_thetas))):
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


def remove_target(peak_thetas, peak_phis, sorted_maxima, target_theta, target_phi, d_theta, d_phi):
    """
    removes target from the peak lists
    :param peak_thetas:
    :param peak_phis:
    :param sorted_maxima:
    :param target_theta:
    :param target_phi:
    :param d_theta:
    :param d_phi:
    :return:
    """
    x, y, z = utils.spherical_to_cartesian_np(1, peak_thetas, peak_phis)
    x_t, y_t, z_t = utils.spherical_to_cartesian_np(1, [target_theta - d_theta / 2, target_theta + d_theta / 2],
                                                    [target_phi - d_phi / 2, target_phi + d_phi / 2])
    x_t = np.sort(x_t)
    y_t = np.sort(y_t)
    z_t = np.sort(z_t)
    to_keep = np.logical_or(x < x_t[0], x > x_t[1]) & np.logical_or(y < y_t[0], y > y_t[1]) & np.logical_or(z < z_t[0],
                                                                                                            z > z_t[1])

    not_to_keep = np.logical_not(to_keep)
    print("Detected target thetas: ", peak_thetas[not_to_keep])
    print("Detected target phis: ", peak_phis[not_to_keep])
    return peak_thetas[to_keep], peak_phis[to_keep], sorted_maxima[to_keep]


def iterative_max_2D_filter(x, r, beamformer, antenna, target_theta=None, target_phi=None,
                            d_theta=None, d_phi=None):
    is_peak_removed = True
    filtered_x = None
    i = 0
    peak_threshold = 0.3
    while is_peak_removed:
        print(f"Iterative max 2d filter, iter: {i}, threshold: {peak_threshold}")
        results, output_signals, thetas, phis = beamformer.compute_beampattern(x=x,
                                                                               N_theta=params.N_theta,
                                                                               N_phi=params.N_phi,
                                                                               fs=params.fs,
                                                                               r=antenna.get_antenna_positions())

        element_beampattern, theta_e, phi_e = antenna.get_antenna_element_beampattern(thetas=thetas,
                                                                                      phis=phis)
        results *= element_beampattern

        filtered_x, is_peak_removed = remove_max_2D(x=x,
                                                    r=r,
                                                    results=results,
                                                    phis=phis,
                                                    thetas=thetas,
                                                    output_signals=output_signals,
                                                    target_theta=target_theta,
                                                    target_phi=target_phi,
                                                    d_theta=d_theta,
                                                    d_phi=d_phi,
                                                    peak_threshold=peak_threshold,
                                                    num_of_removed_signals=1)

        x = filtered_x

        i+=1
    return filtered_x


def remove_max_2D(x, r, results, phis, thetas, output_signals, num_of_removed_signals=None,
                  target_theta=None, target_phi=None, d_theta=None, d_phi=None, peak_threshold=params.peak_threshold):
    """

    :param x: input signal
    :param r: antenna positions
    :param results: beamformer pattern
    :param phis: beamformer pattern phis
    :param thetas: beamformer pattern thetas
    :param output_signals: output signals of beamformer
    :param num_of_removed_signals: number of peaks to remove (Default: None, all the peaks are removed)
    :param target_theta: theta angle (rad) of target
    :param target_phi: phi angle (rad) of target
    :param d_theta: theta range of target
    :param d_phi: phi range of target
    :param peak_threshold: the threshold to count maxima as peak
    :return: fitered_x, is a peak removed
    """
    filtered_x = x.copy()
    N_array = len(r[0])

    # # due to periodicity of phi
    # results = np.vstack((results, results, results))
    # thetas = np.hstack((thetas, thetas, thetas))
    # phis = np.hstack((phis, phis, phis))
    # output_signals = np.vstack((output_signals, output_signals, output_signals))

    maxima = utils.find_relative_maxima(results, threshold=peak_threshold)
    max_val = results[maxima[:, 0], maxima[:, 1]]
    arg_max_val = max_val.argsort()[::-1]
    sorted_maxima = maxima[arg_max_val]

    peak_thetas = thetas[sorted_maxima[:, 1]]
    peak_phis = phis[sorted_maxima[:, 0]]
    filtered_peak_thetas, filtered_peak_phis, sorted_maxima = remove_close_peaks(peak_thetas, peak_phis, sorted_maxima,
                                                                                 eps=0.3)

    if target_theta is not None and target_phi is not None and d_theta is not None and d_phi is not None:
        filtered_peak_thetas, filtered_peak_phis, sorted_maxima = remove_target(filtered_peak_thetas,
                                                                                filtered_peak_phis, sorted_maxima,
                                                                                target_theta, target_phi, d_theta,
                                                                                d_phi)

    if len(filtered_peak_thetas) == 0:
        return filtered_x, False

    if num_of_removed_signals is None:
        num_of_removed_signals = len(filtered_peak_thetas)

    for k in range(0, min(num_of_removed_signals, len(filtered_peak_thetas))):
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
    return filtered_x, True


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

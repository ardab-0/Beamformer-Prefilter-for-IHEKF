import sys
from itertools import combinations
import utils
import numpy as np
from measurement_simulation import compute_phase_shift, compute_phase_shift_near_field
from settings.config import Parameters as params
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt


def remove_close_peaks(theta_peak, phi_peak, sorted_maxima, eps=0.1):
    """
    removes peaks that are closer than eps
    needs further testing

    :param sorted_maxima: sorted maxima indices
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
    """
    removes all the peaks from the input signal x and returns filtered signal
    :param x:
    :param r:
    :param results:
    :param phis:
    :param thetas:
    :param output_signals:
    :param peak_threshold:
    :return:
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


def remove_target_with_sidelobes(peak_thetas, peak_phis, sorted_maxima, beamformer, antenna, target_theta, target_phi,
                                 eps, peak_threshold):
    """
    remove target peaks from the peaks list (also considers the sidelobes and removes them)
    :param peak_thetas:
    :param peak_phis:
    :param sorted_maxima:
    :param beamformer:
    :param antenna:
    :param target_theta:
    :param target_phi:
    :param eps:
    :param peak_threshold:
    :return:
    """
    r = antenna.get_antenna_positions()
    u = np.array(
        [np.sin(target_theta) * np.cos(target_phi), np.sin(target_theta) * np.sin(target_phi), np.cos(target_theta)])
    k = -2 * np.pi * params.f / params.c * u
    a_k = np.exp(-1j * k.T @ r).reshape((-1, 1))

    results, output_signals, thetas, phis = beamformer.compute_beampattern(x=a_k,
                                                                           N_theta=params.N_theta,
                                                                           N_phi=params.N_phi,
                                                                           fs=params.fs,
                                                                           r=r)

    maxima = utils.find_relative_maxima(results,
                                        threshold=0.3)  #####################################################################################################################################
    max_val = results[maxima[:, 0], maxima[:, 1]]
    arg_max_val = max_val.argsort()[::-1]
    sorted_maxima_ideal = maxima[arg_max_val]

    ideal_peak_thetas = thetas[sorted_maxima_ideal[:, 1]]
    ideal_peak_phis = phis[sorted_maxima_ideal[:, 0]]
    x_ideal, y_ideal, z_ideal = utils.spherical_to_cartesian_np(1, ideal_peak_thetas, ideal_peak_phis)

    x, y, z = utils.spherical_to_cartesian_np(1, peak_thetas, peak_phis)

    dx = x.reshape((-1, 1)) - x_ideal.reshape((1, -1))
    dy = y.reshape((-1, 1)) - y_ideal.reshape((1, -1))
    dz = z.reshape((-1, 1)) - z_ideal.reshape((1, -1))
    d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    matching_peaks = d <= eps
    atleast_one_matching_peak = np.sum(matching_peaks.astype(int), axis=1)
    idx_to_keep = atleast_one_matching_peak == 0

    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # ax.set_xlim(params.room_x)
    # ax.set_ylim(params.room_y)
    # ax.set_zlim(params.room_z)
    #
    # ax.set_xlabel("x(m)")
    # ax.set_ylabel("y(m)")
    # ax.set_zlabel("z(m)")
    #
    # beampattern_cartesian = beamformer.spherical_to_cartesian(results, thetas=thetas, phis=phis)
    # beampattern_cartesian = beampattern_cartesian + antenna.get_t()  # place the pattern on antenna position
    #
    # ax.scatter3D(beampattern_cartesian[0, :], beampattern_cartesian[1, :], beampattern_cartesian[2, :])
    #
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #
    # theta, phi = np.meshgrid(thetas, phis)
    #
    # surf = ax.plot_surface(theta, phi, results,
    #                        linewidth=0, antialiased=False)
    # ax.set_xlabel("theta (rad)")
    # ax.set_ylabel("phi (rad)")
    # ax.set_zlabel("power")
    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
    return peak_thetas[idx_to_keep], peak_phis[idx_to_keep], sorted_maxima[idx_to_keep]


def keep_target(peak_thetas, peak_phis, sorted_maxima, target_theta, target_phi, cone_angle):
    """
    only keeps target in the peak lists

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

    to_keep = utils.cone_filter(np.stack([x, y, z]), target_theta=target_theta, target_phi=target_phi,
                                cone_angle=cone_angle)

    print("Detected target thetas: ", peak_thetas[to_keep])
    print("Detected target phis: ", peak_phis[to_keep])
    return peak_thetas[to_keep], peak_phis[to_keep], sorted_maxima[to_keep]


def remove_target(peak_thetas, peak_phis, sorted_maxima, target_theta, target_phi, cone_angle):
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

    not_to_keep = utils.cone_filter(np.stack([x, y, z]), target_theta=target_theta, target_phi=target_phi,
                                    cone_angle=cone_angle)

    to_keep = np.logical_not(not_to_keep)
    print("Detected target thetas: ", peak_thetas[not_to_keep])
    print("Detected target phis: ", peak_phis[not_to_keep])
    return peak_thetas[to_keep], peak_phis[to_keep], sorted_maxima[to_keep]


def two_step_filter(x, r, num_of_removed_signals=None,
                    target_theta=None, target_phi=None, cone_angle=None, peak_threshold=params.peak_threshold,
                    beamformer=None, antenna=None, target_position=None):
    """
    removes the target signal from the original signal, then removes remaining signal from the original signal

    :param eps:
    :param beamformer: used beamformer
    :param antenna: current antenna array
    :param x: input signal
    :param r: antenna positions
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
    results, output_signals, thetas, phis = beamformer.compute_beampattern(x=x,
                                                                           N_theta=params.N_theta,
                                                                           N_phi=params.N_phi,
                                                                           fs=params.fs,
                                                                           r=antenna.get_antenna_positions())

    element_beampattern, theta_e, phi_e = antenna.get_antenna_element_beampattern(thetas=thetas,
                                                                                  phis=phis)
    results *= element_beampattern

    maxima = utils.find_relative_maxima(results, threshold=peak_threshold)
    max_val = results[maxima[:, 0], maxima[:, 1]]
    arg_max_val = max_val.argsort()[::-1]
    sorted_maxima = maxima[arg_max_val]

    peak_thetas = thetas[sorted_maxima[:, 1]]
    peak_phis = phis[sorted_maxima[:, 0]]

    if target_theta is not None and target_phi is not None and cone_angle is not None:
        # without sidelobe
        # peak_thetas, peak_phis, sorted_maxima = remove_close_peaks(peak_thetas, peak_phis, sorted_maxima, eps=0.3)
        peak_thetas, peak_phis, sorted_maxima = keep_target(peak_thetas,
                                                            peak_phis, sorted_maxima,
                                                            target_theta, target_phi, cone_angle=cone_angle
                                                            )
        # # with sidelobe
        # peak_thetas, peak_phis, sorted_maxima = remove_target_with_sidelobes(peak_thetas, peak_phis, sorted_maxima, beamformer, antenna, target_theta,
        #                              target_phi, 0.1, peak_threshold)

    if len(peak_thetas) == 0:
        return filtered_x, False

    if num_of_removed_signals is None:
        num_of_removed_signals = len(peak_thetas)

    for k in range(0, min(num_of_removed_signals, len(peak_thetas))):
        theta_to_remove = peak_thetas[k]
        phi_to_remove = peak_phis[k]
        print(f"theta and phi to remove (rad): {theta_to_remove}, {phi_to_remove}")

        u = np.array(
            [np.sin(theta_to_remove) * np.cos(phi_to_remove), np.sin(theta_to_remove) * np.sin(phi_to_remove),
             np.cos(theta_to_remove)])
        signal_to_remove = output_signals[sorted_maxima[k][0], sorted_maxima[k][1]]


        t = np.arange(params.N) / params.fs
        z = np.exp(1j * (2 * np.pi * params.f * (t.reshape((1, -1))) - np.pi / 2)).reshape(-1)
        # plt.figure()
        # plt.plot(signal_to_remove, label="beamformer")
        # plt.plot(z, label="z")
        # plt.legend()

        signal_to_remove_at_antenna = np.zeros((N_array, params.N), dtype=complex)
        for i in range(N_array):
            if target_position is not None:
                signal_to_remove_at_antenna[i, :] = compute_phase_shift_near_field(signal_to_remove, params.f, r[:, i],
                                                                                   target_position)
            else:
                signal_to_remove_at_antenna[i, :] = compute_phase_shift(signal_to_remove, params.f, u, r[:, i])


        filtered_x -= signal_to_remove_at_antenna

    filtered_x = x - filtered_x

    return filtered_x, True


def ground_reflection_filter(x, r, target_x, target_y, target_z, cone_angle, peak_threshold, beamformer, antenna, reflection_position=None):
    reflection_dir = np.array([target_x, target_y, -target_z]).reshape((-1, 1)) - antenna.get_t()
    reflection_r, reflection_theta, reflection_phi = utils.cartesian_to_spherical(reflection_dir[0], reflection_dir[1],
                                                                                  reflection_dir[2])
    filtered_x = x.copy()
    N_array = len(r[0])
    results, output_signals, thetas, phis = beamformer.compute_beampattern(x=x,
                                                                           N_theta=params.N_theta,
                                                                           N_phi=params.N_phi,
                                                                           fs=params.fs,
                                                                           r=antenna.get_antenna_positions())

    element_beampattern, theta_e, phi_e = antenna.get_antenna_element_beampattern(thetas=thetas,
                                                                                  phis=phis)
    results *= element_beampattern

    maxima = utils.find_relative_maxima(results, threshold=peak_threshold)
    max_val = results[maxima[:, 0], maxima[:, 1]]
    arg_max_val = max_val.argsort()[::-1]
    sorted_maxima = maxima[arg_max_val]

    peak_thetas = thetas[sorted_maxima[:, 1]]
    peak_phis = phis[sorted_maxima[:, 0]]

    peak_thetas, peak_phis, sorted_maxima = keep_target(peak_thetas,
                                                        peak_phis,
                                                        sorted_maxima,
                                                        reflection_theta,
                                                        reflection_phi,
                                                        cone_angle=cone_angle
                                                        )

    if len(peak_thetas) == 0:
        return filtered_x

    theta_to_remove = peak_thetas[0]
    phi_to_remove = peak_phis[0]
    print(f"theta and phi to remove (rad): {theta_to_remove}, {phi_to_remove}")

    u = np.array(
        [np.sin(theta_to_remove) * np.cos(phi_to_remove), np.sin(theta_to_remove) * np.sin(phi_to_remove),
         np.cos(theta_to_remove)])
    signal_to_remove = output_signals[sorted_maxima[0][0], sorted_maxima[0][1]]

    signal_to_remove_at_antenna = np.zeros((N_array, params.N), dtype=complex)
    for i in range(N_array):
        if reflection_position is not None:
            signal_to_remove_at_antenna[i, :] = compute_phase_shift_near_field(signal_to_remove, params.f, r[:, i],
                                                                               reflection_position)
        else:
            signal_to_remove_at_antenna[i, :] = compute_phase_shift(signal_to_remove, params.f, u, r[:, i])

    filtered_x -= signal_to_remove_at_antenna
    return filtered_x


def multipath_search_filter(x, r, beamformer, antenna, peak_threshold, target_theta, target_phi,
                            cone_angle):
    """
    tries to remove signals as long as target signal is not
    filtered (currently iteration stops when a signal source
    is eliminated and target signal is intact, success of filtering is not checked)

    :param x:
    :param r:
    :param beamformer:
    :param antenna:
    :param peak_threshold:
    :param target_theta:
    :param target_phi:
    :param d_theta:
    :param d_phi:
    :return:
    """
    N_array = len(r[0])
    filtered_x = x.copy()
    results, output_signals, thetas, phis = beamformer.compute_beampattern(x=x,
                                                                           N_theta=params.N_theta,
                                                                           N_phi=params.N_phi,
                                                                           fs=params.fs,
                                                                           r=antenna.get_antenna_positions())

    element_beampattern, theta_e, phi_e = antenna.get_antenna_element_beampattern(thetas=thetas,
                                                                                  phis=phis)
    results *= element_beampattern

    maxima = utils.find_relative_maxima(results, threshold=peak_threshold)
    max_val = results[maxima[:, 0], maxima[:, 1]]
    arg_max_val = max_val.argsort()[::-1]
    sorted_maxima = maxima[arg_max_val]

    peak_thetas = thetas[sorted_maxima[:, 1]]
    peak_phis = phis[sorted_maxima[:, 0]]

    target_thetas, target_phis, target_sorted_maxima = keep_target(peak_thetas,
                                                                   peak_phis,
                                                                   sorted_maxima,
                                                                   target_theta,
                                                                   target_phi,
                                                                   cone_angle=cone_angle
                                                                   )

    if len(target_thetas) == 0:
        raise ValueError("Target is not detected")

    non_target_thetas, non_target_phis, non_target_sorted_maxima = remove_target(peak_thetas,
                                                                                 peak_phis,
                                                                                 sorted_maxima,
                                                                                 target_theta,
                                                                                 target_phi,
                                                                                 cone_angle,
                                                                                 )
    target = {"theta": target_thetas[0],
              "phi": target_phis[0],
              "maxima": target_sorted_maxima[0],
              "amplitude": results[target_sorted_maxima[0, 0], target_sorted_maxima[0, 1]]}
    for i in range(len(non_target_thetas)):
        filtered_x = x.copy()
        theta_to_remove = non_target_thetas[i]
        phi_to_remove = non_target_phis[i]
        print(f"theta and phi to remove (rad): {theta_to_remove}, {phi_to_remove}")

        u = np.array(
            [np.sin(theta_to_remove) * np.cos(phi_to_remove), np.sin(theta_to_remove) * np.sin(phi_to_remove),
             np.cos(theta_to_remove)])
        signal_to_remove = output_signals[non_target_sorted_maxima[i][0], non_target_sorted_maxima[i][1]]

        signal_to_remove_at_antenna = np.zeros((N_array, params.N), dtype=complex)
        for i in range(N_array):
            signal_to_remove_at_antenna[i, :] = compute_phase_shift(signal_to_remove, params.f, u, r[:, i])
        filtered_x -= signal_to_remove_at_antenna

        results, output_signals, thetas, phis = beamformer.compute_beampattern(x=filtered_x,
                                                                               N_theta=params.N_theta,
                                                                               N_phi=params.N_phi,
                                                                               fs=params.fs,
                                                                               r=antenna.get_antenna_positions())

        results *= element_beampattern

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        theta, phi = np.meshgrid(thetas, phis)

        surf = ax.plot_surface(theta, phi, results,
                               linewidth=0, antialiased=False)
        ax.set_xlabel("theta (rad)")
        ax.set_ylabel("phi (rad)")
        ax.set_zlabel("power")
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

        maxima = utils.find_relative_maxima(results, threshold=peak_threshold)
        max_val = results[maxima[:, 0], maxima[:, 1]]
        arg_max_val = max_val.argsort()[::-1]
        sorted_maxima = maxima[arg_max_val]

        peak_thetas = thetas[sorted_maxima[:, 1]]
        peak_phis = phis[sorted_maxima[:, 0]]

        target_thetas, target_phis, target_sorted_maxima = keep_target(peak_thetas,
                                                                       peak_phis,
                                                                       sorted_maxima,
                                                                       target_theta,
                                                                       target_phi,
                                                                       cone_angle=cone_angle,
                                                                       )

        if len(target_thetas) == 0:
            continue
        else:
            if np.abs(target["amplitude"] - results[target_sorted_maxima[0, 0], target_sorted_maxima[0, 1]]) < 0.2:
                break

    return filtered_x


def iterative_max_2D_filter(x, r, beamformer, antenna, peak_threshold, target_theta=None, target_phi=None,
                            cone_angle=None, max_iteration=sys.maxsize):
    """
    removes the peaks iteratively while keeping the target peaks intact

    :param x:
    :param r:
    :param beamformer:
    :param antenna:
    :param peak_threshold:
    :param target_theta:
    :param target_phi:
    :param d_theta:
    :param d_phi:
    :param max_iteration:
    :return:
    """
    is_peak_removed = True
    filtered_x = None
    i = 0
    while is_peak_removed and i < max_iteration:
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
                                                    cone_angle=cone_angle,
                                                    peak_threshold=peak_threshold,
                                                    num_of_removed_signals=1,
                                                    beamformer=beamformer,
                                                    antenna=antenna)

        x = filtered_x

        i += 1
    return filtered_x


def remove_max_2D(x, r, results, phis, thetas, output_signals, num_of_removed_signals=None,
                  target_theta=None, target_phi=None, cone_angle=None, peak_threshold=params.peak_threshold,
                  beamformer=None, antenna=None, eps=0.1):
    """
    removes the maximum k (num_of_removed_signals) peaks from the signal simultaneously

    :param eps:
    :param beamformer: used beamformer
    :param antenna: current antenna array
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

    if target_theta is not None and target_phi is not None and cone_angle is not None:
        # without sidelobe
        peak_thetas, peak_phis, sorted_maxima = remove_close_peaks(peak_thetas, peak_phis, sorted_maxima, eps=0.3)
        peak_thetas, peak_phis, sorted_maxima = remove_target(peak_thetas,
                                                              peak_phis, sorted_maxima,
                                                              target_theta, target_phi, cone_angle,
                                                              )
        # # with sidelobe
        # peak_thetas, peak_phis, sorted_maxima = remove_target_with_sidelobes(peak_thetas, peak_phis, sorted_maxima, beamformer, antenna, target_theta,
        #                              target_phi, 0.1, peak_threshold)

    if len(peak_thetas) == 0:
        return filtered_x, False

    if num_of_removed_signals is None:
        num_of_removed_signals = len(peak_thetas)

    for k in range(0, min(num_of_removed_signals, len(peak_thetas))):
        theta_to_remove = peak_thetas[k]
        phi_to_remove = peak_phis[k]
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


def multipath_filter(x, r, beamformer, antenna, peak_threshold, target_theta, target_phi, cone_angle,
                     ):
    filtered_x = x.copy()
    N_array = len(r[0])
    results, output_signals, thetas, phis = beamformer.compute_beampattern(x=x,
                                                                           N_theta=params.N_theta,
                                                                           N_phi=params.N_phi,
                                                                           fs=params.fs,
                                                                           r=antenna.get_antenna_positions())

    element_beampattern, theta_e, phi_e = antenna.get_antenna_element_beampattern(thetas=thetas,
                                                                                  phis=phis)
    results *= element_beampattern

    maxima = utils.find_relative_maxima(results, threshold=peak_threshold)
    max_val = results[maxima[:, 0], maxima[:, 1]]
    arg_max_val = max_val.argsort()[::-1]
    sorted_maxima = maxima[arg_max_val]

    peak_thetas = thetas[sorted_maxima[:, 1]]
    peak_phis = phis[sorted_maxima[:, 0]]

    target_peak_thetas, target_peak_phis, target_sorted_maxima = keep_target(peak_thetas,
                                                                             peak_phis,
                                                                             sorted_maxima,
                                                                             target_theta,
                                                                             target_phi,
                                                                             cone_angle=cone_angle,
                                                                             )

    non_target_peak_thetas, non_target_peak_phis, non_target_sorted_maxima = remove_target(peak_thetas,
                                                                                           peak_phis,
                                                                                           sorted_maxima,
                                                                                           target_theta,
                                                                                           target_phi,
                                                                                           cone_angle=cone_angle,
                                                                                           )

    detected_target_theta = target_peak_thetas[0]
    detected_target_phi = target_peak_phis[0]
    detected_target_amplitude = results[target_sorted_maxima[0, 0], target_sorted_maxima[0, 1]]

    u = np.array(
        [np.sin(detected_target_theta) * np.cos(detected_target_phi),
         np.sin(detected_target_theta) * np.sin(detected_target_phi), np.cos(detected_target_theta)])
    k = -2 * np.pi * params.f / params.c * u
    a_k = detected_target_amplitude * np.exp(-1j * k.T @ r).reshape((-1, 1))

    reference_beampattern, _, _, _ = beamformer.compute_beampattern(x=a_k,
                                                                    N_theta=params.N_theta,
                                                                    N_phi=params.N_phi,
                                                                    fs=params.fs,
                                                                    r=r)
    errors = []
    filtered_xs = []
    for i in range(len(non_target_peak_thetas)):
        filtered_x = x.copy()
        theta_to_remove = non_target_peak_thetas[i]
        phi_to_remove = non_target_peak_phis[i]
        print(f"theta and phi to remove (rad): {theta_to_remove}, {phi_to_remove}")

        u = np.array(
            [np.sin(theta_to_remove) * np.cos(phi_to_remove), np.sin(theta_to_remove) * np.sin(phi_to_remove),
             np.cos(theta_to_remove)])
        signal_to_remove = output_signals[non_target_sorted_maxima[i][0], non_target_sorted_maxima[i][1]]

        signal_to_remove_at_antenna = np.zeros((N_array, params.N), dtype=complex)
        for i in range(N_array):
            signal_to_remove_at_antenna[i, :] = compute_phase_shift(signal_to_remove, params.f, u, r[:, i])
        filtered_x -= signal_to_remove_at_antenna
        filtered_xs.append(filtered_x)

        results, output_signals, thetas, phis = beamformer.compute_beampattern(x=filtered_x,
                                                                               N_theta=params.N_theta,
                                                                               N_phi=params.N_phi,
                                                                               fs=params.fs,
                                                                               r=antenna.get_antenna_positions())

        error = np.sqrt(np.mean((reference_beampattern - results) ** 2))
        errors.append(error)
        # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        #
        # theta, phi = np.meshgrid(thetas, phis)
        #
        # surf = ax.plot_surface(theta, phi, results,
        #                        linewidth=0, antialiased=False)
        # ax.set_xlabel("theta (rad)")
        # ax.set_ylabel("phi (rad)")
        # ax.set_zlabel("power")
        # # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.show()
    if len(errors) == 0:
        return x
    errors = np.array(errors)
    print(errors)
    min_idx = np.argmin(errors)
    return filtered_xs[min_idx]

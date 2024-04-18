import os.path
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.linalg
from tqdm import tqdm
import spatial_filter
import utils
from antenna_array import SimulationAntennaArray
from beamformer.beamformer import generate_beamformer
from real_data import plot_2d
from settings.antenna_element_positions import generate_antenna_element_positions
from jacobian import Jacobian_h, jacobian_numpy
import matplotlib.pyplot as plt
from settings.config import Parameters
from matplotlib import cm
import measurement_simulation as sim
from settings.config import VERBOSE


def simulate(params):
    spatial_filter_collection = spatial_filter.SpatialFilter(params=params)

    antenna_element_positions, A_full = generate_antenna_element_positions(kind=params.antenna_kind, lmb=params.lmb,
                                                                           get_A_full=True)
    antenna_element_positions[[0, 1], :] = antenna_element_positions[[1, 0], :]  # switch x and y rows

    beacon_pos = utils.generate_spiral_path(a=1, theta_extent=10, alpha=np.pi / 45)

    ant1 = SimulationAntennaArray(rot=[0, 45, 45], t=[-2, -3, 3], element_positions=antenna_element_positions)
    ant2 = SimulationAntennaArray(rot=[0, 45, 200], t=[4, 2, 3], element_positions=antenna_element_positions)
    ant3 = SimulationAntennaArray(rot=[0, 45, -60], t=[-2, 3, 3], element_positions=antenna_element_positions)
    antenna_list = [ant1, ant2, ant3]

    # initial state
    xs = []
    x = np.array([[beacon_pos[0, 0], beacon_pos[1, 0], beacon_pos[2, 0], 0, 0, 0]]).T
    sigma = np.eye(len(x))  # try individual values
    jacobian_cache = {}
    t = np.arange(params.N) / params.fs
    beamformer = generate_beamformer(beamformer_type=params.beamformer_type)
    recorded_phi_differences = []
    baseleine_phi_differences = []

    for k in tqdm(range(len(beacon_pos[0]))):
        # prediction
        G = sim.compute_G(params.dt)
        Q = params.sigma_a ** 2 * G @ G.T
        F = sim.compute_F(params.dt)
        x = F @ x
        sigma = F @ sigma @ F.T + Q
        phi_B = np.random.rand() * 2 * np.pi  # transmitter phase at time k
        # multipath_sources = sim.generate_multipath_sources(multipath_count=params.multipath_count)
        multipath_sources = sim.generate_multipath_sources_with_amplitude(multipath_count=params.multipath_count, amplitude=params.multipath_amplitude)
        # multipath_sources = [
        #     {"x": beacon_pos[0, k],
        #      "y": beacon_pos[1, k],
        #      "z": -beacon_pos[2, k],
        #      "a": 0.90 + np.random.randn(1) * 0.1}]
        # iteration
        x_0 = x
        for i in params.i_list:
            ant_pos_i = []
            A = []
            R = []
            h = []
            phi = []
            phi_ref = []

            # to visualize beam pattern at each step
            cartesian_beampattern_list = []
            beampattern_2d_list = []

            for antenna in antenna_list:
                ant_pos = antenna.get_antenna_positions()
                ant_pos_m_i = ant_pos[:, : i]
                ant_pos_i.append(ant_pos_m_i)

                # estimate target dir with respect to antenna from estimated position x_0
                target_dir = x_0[:3].reshape((-1, 1)) - antenna.get_t()
                target_dir_r, target_dir_theta, target_dir_phi = utils.cartesian_to_spherical(target_dir[0],
                                                                                              target_dir[1],
                                                                                              target_dir[2])

                real_target_dir = beacon_pos[:, k].reshape((-1, 1)) - antenna.get_t()
                real_target_dir_r, real_target_dir_theta, real_target_dir_phi = utils.cartesian_to_spherical(
                    real_target_dir[0], real_target_dir[1],
                    real_target_dir[2])
                if VERBOSE:
                    print(f"Target direction theta: {real_target_dir_theta}, phi: {real_target_dir_phi}")
                    for multipath in multipath_sources:
                        dir = np.array([multipath["x"], multipath["y"], multipath["z"]]).reshape((-1, 1)) - antenna.get_t()
                        m_r, m_t, m_p = utils.cartesian_to_spherical(dir[0], dir[1], dir[2])
                        print(f"Multipath direction theta: {m_t}, phi: {m_p}")

                # antennas_used_in_beamformer = params.i_list[0]
                antennas_used_in_beamformer = i
                if params.use_multipath:

                    if params.measure_phi_m_directly:
                        phi_m, multipath_sources_at_k = sim.measure_multipath(ant_pos,
                                                                              beacon_pos[:, k].reshape((-1, 1)),
                                                                              sigma_phi=params.sigma_phi,
                                                                              multipath_count=params.multipath_count)
                    else:
                        s_m = sim.measure_s_m_multipath(t=t, antenna_positions=ant_pos,
                                                        beacon_pos=beacon_pos[:, k].reshape((-1, 1)),
                                                        phi_B=phi_B, sigma=params.sigma,
                                                        multipath_sources=multipath_sources)

                        if params.apply_element_pattern or params.visualize_beampatterns or params.apply_spatial_filter:
                            results, output_signals, thetas, phis = beamformer.compute_beampattern(
                                x=s_m[:antennas_used_in_beamformer],
                                N_theta=params.N_theta,
                                N_phi=params.N_phi,
                                fs=params.fs,
                                r=ant_pos[:, :antennas_used_in_beamformer])

                        if params.apply_element_pattern:
                            element_beampattern, theta_e, phi_e = antenna.get_antenna_element_beampattern(thetas=thetas,
                                                                                                          phis=phis)
                            results *= element_beampattern

                        if params.visualize_beampatterns:
                            # x = np.abs(element_beampattern) * np.sin(theta_e) * np.cos(phi_e)
                            # y = np.abs(element_beampattern) * np.sin(theta_e) * np.sin(phi_e)
                            # z = np.abs(element_beampattern) * np.cos(theta_e)
                            #
                            # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                            # # ax.scatter3D(x, y, z)
                            # ax.plot_surface(x, y, z, rstride=2, cstride=2, color='white',
                            #                 shade=False, edgecolor='k')
                            # ax.set_xlim([-1.5, 1.5])
                            # ax.set_ylim([-1.5, 1.5])
                            # ax.set_zlim([-1.5, 1.5])
                            # plt.show()

                            # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                            # ax.plot_surface(theta_e, phi_e, element_beampattern, vmin=element_beampattern.min() * 2, cmap=cm.Blues)
                            # plt.show()

                            beampattern_2d_list.append({"results": results,
                                                        "thetas": thetas,
                                                        "phis": phis,
                                                        "ant_pos": ant_pos_m_i,
                                                        "s_m": s_m,
                                                        "output_signals": output_signals})
                            beampattern_cartesian = beamformer.spherical_to_cartesian(results, thetas=thetas, phis=phis)
                            beampattern_cartesian = beampattern_cartesian + antenna.get_t()  # place the pattern on antenna position
                            cartesian_beampattern_list.append(beampattern_cartesian)

                        if params.apply_spatial_filter and k >= params.spatial_filter_initialization_index:
                            # s_m = spatial_filter_collection.remove_components_2D(x=s_m, r=ant_pos,
                            #                            results=results, phis=phis, thetas=thetas,
                            #                            output_signals=output_signals)
                            s_m = spatial_filter_collection.iterative_max_2D_filter(x=s_m,
                                                                                  r=ant_pos,
                                                                                  beamformer=beamformer,
                                                                                  antenna=antenna,
                                                                                  peak_threshold=params.peak_threshold,
                                                                                  target_theta=target_dir_theta,
                                                                                  target_phi=target_dir_phi,
                                                                                  cone_angle=np.deg2rad(
                                                                                      params.cone_angle),
                                                                                  max_iteration=params.max_iteration) # needs to be adaptive
                            # s_m, _ = spatial_filter_collection.two_step_filter(x=s_m,
                            #                                         r=ant_pos,
                            #                                         beamformer=beamformer,
                            #                                         antenna=antenna,
                            #                                         peak_threshold=0.1,
                            #                                         target_theta=target_dir_theta,
                            #                                         target_phi=target_dir_phi,
                            #                                         cone_angle=np.deg2rad(
                            #                                             params.cone_angle),
                            #                                         num_of_removed_signals=1,
                            #                                         antennas_used_in_beamformer=antennas_used_in_beamformer,
                            #                                         # uses only the first iteration's antennas in beamformer
                            #                                         # target_position=x_0[:3].reshape(-1)
                            #                                         )

                            # s_m = spatial_filter_collection.multipath_filter(x=s_m,
                            #                                       r=ant_pos,
                            #                                       beamformer=beamformer,
                            #                                       antenna=antenna,
                            #                                       peak_threshold=0.3,
                            #                                       target_theta=target_dir_theta,
                            #                                       target_phi=target_dir_phi,
                            #                                       d_theta=np.deg2rad(
                            #                                           params.target_theta_range_deg),
                            #                                       d_phi=np.deg2rad(params.target_phi_range_deg))

                            # s_m = spatial_filter_collection.ground_reflection_filter(x=s_m,
                            #                                               r=ant_pos,
                            #                                               beamformer=beamformer,
                            #                                               antenna=antenna,
                            #                                               peak_threshold=0.1,
                            #                                               target_x=x_0[0],
                            #                                               target_y=x_0[1],
                            #                                               target_z=x_0[2],
                            #                                               cone_angle=np.deg2rad(params.cone_angle),
                            #                                               )

                            if params.visualize_beampatterns:
                                results, output_signals, thetas, phis = beamformer.compute_beampattern(
                                    x=s_m[:antennas_used_in_beamformer],
                                    N_theta=params.N_theta,
                                    N_phi=params.N_phi,
                                    fs=params.fs,
                                    r=ant_pos[:, :antennas_used_in_beamformer])
                                # results = np.sqrt(results) # power to amplitude conversion
                                beampattern_2d_list[-1].update({"results_filtered": results,
                                                                "thetas_filtered": thetas,
                                                                "phis_filtered": phis,
                                                                "ant_pos_filtered": ant_pos,
                                                                "s_m_filtered": s_m,
                                                                "output_signals_filtered": output_signals})

                        phi_m = sim.measure_phi(s_m=s_m)


                else:
                    if params.measure_phi_m_directly:
                        phi_m = sim.measure(ant_pos, beacon_pos[:, k].reshape((-1, 1)), sigma_phi=params.sigma_phi)
                    else:
                        s_m = sim.measure_s_m(t=t, antenna_positions=ant_pos,
                                              beacon_pos=beacon_pos[:, k].reshape((-1, 1)),
                                              phi_B=phi_B, sigma=params.sigma)
                        phi_m = sim.measure_phi(s_m)

                    if params.visualize_beampatterns:
                        results, output_signals, thetas, phis = beamformer.compute_beampattern(
                            x=s_m[:antennas_used_in_beamformer],
                            N_theta=params.N_theta,
                            N_phi=params.N_phi,
                            fs=params.fs,
                            r=ant_pos[:, :antennas_used_in_beamformer])
                        # results = np.sqrt(results) # power to amplitude conversion
                        beampattern_2d_list.append({"results": results,
                                                    "thetas": thetas,
                                                    "phis": phis,
                                                    "ant_pos": ant_pos,
                                                    "s_m": s_m,
                                                    "output_signals": output_signals})
                        beampattern_cartesian = beamformer.spherical_to_cartesian(results, thetas=thetas, phis=phis)
                        beampattern_cartesian = beampattern_cartesian + antenna.get_t()  # place the pattern on antenna position
                        cartesian_beampattern_list.append(beampattern_cartesian)

                phi.append(phi_m[: i])

                phi_m_ref = sim.measure(ant_pos_m_i, beacon_pos[:, k].reshape((-1, 1)), sigma_phi=0)
                phi_ref.append(phi_m_ref[: i])

                A_m = A_full[: i - 1, : i]
                A.append(A_m.tolist())

                R_m = params.sigma_phi ** 2 * A_m @ A_m.T
                R.append(R_m.tolist())

                h_m = sim.measure(ant_pos_m_i, x[:3].reshape((-1, 1)), sigma_phi=0)
                h.append(h_m)

            ################################### visualize beampatterns
            if params.visualize_beampatterns:
                fig = plt.figure()
                ax = plt.axes(projection="3d")
                ax.set_xlim(params.room_x)
                ax.set_ylim(params.room_y)
                ax.set_zlim(params.room_z)

                ax.set_xlabel("x(m)")
                ax.set_ylabel("y(m)")
                ax.set_zlabel("z(m)")
                for beampattern in cartesian_beampattern_list:
                    ax.scatter3D(beampattern[0, :], beampattern[1, :], beampattern[2, :])
                ax.plot3D(beacon_pos[0, :], beacon_pos[1, :], beacon_pos[2, :], "green")
                ax.scatter3D(beacon_pos[0, k], beacon_pos[1, k], beacon_pos[2, k], c="red")

                for multipath in multipath_sources:
                    ax.scatter3D(multipath["x"], multipath["y"], multipath["z"], "red")

                fig, ax = plt.subplots(2, 2, subplot_kw={"projection": "3d"})
                for i, beampattern_2d in enumerate(beampattern_2d_list):
                    plt.subplot(2, 2, i + 1)
                    thetas = beampattern_2d["thetas"]
                    phis = beampattern_2d["phis"]
                    thetas, phis = np.meshgrid(thetas, phis)
                    r = beampattern_2d["results"]
                    surf = ax[i // 2, i % 2].plot_surface(thetas, phis, r, cmap=cm.coolwarm,
                                                          linewidth=0, antialiased=False)
                    ax[i // 2, i % 2].set_xlabel("theta (rad)")
                    ax[i // 2, i % 2].set_ylabel("phi (rad)")
                    ax[i // 2, i % 2].set_zlabel("power")
                    # Add a color bar which maps values to colors.
                    fig.colorbar(surf, shrink=0.5, aspect=5)

                if params.apply_spatial_filter:
                    fig, ax = plt.subplots(2, 2, subplot_kw={"projection": "3d"})
                    for i, beampattern_2d in enumerate(beampattern_2d_list):
                        plt.subplot(2, 2, i + 1)
                        thetas = beampattern_2d["thetas_filtered"]
                        phis = beampattern_2d["phis_filtered"]
                        thetas, phis = np.meshgrid(thetas, phis)
                        r = beampattern_2d["results_filtered"]
                        surf = ax[i // 2, i % 2].plot_surface(thetas, phis, r,
                                                              cmap=cm.coolwarm,
                                                              linewidth=0, antialiased=False)
                        ax[i // 2, i % 2].set_xlabel("theta (rad)")
                        ax[i // 2, i % 2].set_ylabel("phi (rad)")
                        ax[i // 2, i % 2].set_zlabel("power")
                        # Add a color bar which maps values to colors.
                        fig.colorbar(surf, shrink=0.5, aspect=5)
                plt.show()
            ################################### visualize beampatterns

            ant_pos_i = np.hstack(ant_pos_i)
            phi = np.vstack(phi)
            phi_ref = np.vstack(phi_ref)
            A = scipy.linalg.block_diag(*A)
            R = scipy.linalg.block_diag(*R)
            h = np.vstack(h)

            z = A @ phi
            h = utils.mod_2pi(A @ h)
            recorded_phi_differences.append(utils.mod_2pi(A @ phi))
            baseleine_phi_differences.append(utils.mod_2pi(A @ phi_ref))

            if params.jacobian_type == "scipy":
                if i not in jacobian_cache:
                    # use scipy implementation of jacobian (slow)
                    h_jacobian = Jacobian_h(N=A.shape[0], I=A.shape[1], w0=2 * np.pi * params.f, c=params.c)
                    jacobian_cache[i] = h_jacobian
                    h_jacobian.compute_jacobian()
                    H = h_jacobian.evaluate_jacobian(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0], ant_pos=ant_pos_i)
                else:
                    H = jacobian_cache[i].evaluate_jacobian(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0],
                                                            ant_pos=ant_pos_i)
            elif params.jacobian_type == "numpy":
                H = jacobian_numpy(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0], ant_pos=ant_pos_i, c=params.c,
                                   w0=2 * np.pi * params.f)
            else:
                raise ValueError('jacobian_type is incorrect.')

            K = sigma @ H.T @ np.linalg.inv(R + H @ sigma @ H.T)
            res = utils.mod_2pi(z - h - H @ (
                    x_0 - x))
            x = x + K @ res  ################################################## in paper: x_0 - x

        # update
        sigma = (np.eye(len(x)) - K @ H) @ sigma

        # print("Beacon position: \n", beacon_pos[0:3, k])
        # print("x_0: \n", x_0)
        # print("x: \n", x)
        # print("\n\n")
        xs.append(x)

    xs = np.array(xs).squeeze()
    return xs, beacon_pos, antenna_list, recorded_phi_differences, baseleine_phi_differences

def main(params):

    xs, beacon_pos, antenna_list, recorded_phi_differences, baseleine_phi_differences = simulate(params)
    # creating an empty canvas
    fig = plt.figure()

    ax = plt.axes(projection="3d")
    ax.set_xlim(params.room_x)
    ax.set_ylim(params.room_y)
    ax.set_zlim(params.room_z)

    ax.set_xlabel("x(m)")
    ax.set_ylabel("y(m)")
    ax.set_zlabel("z(m)")

    # ax.plot3D(xs[:, 0], xs[:, 1], xs[:, 2], 'red')
    ax.plot3D(xs[:, 0], xs[:, 1], xs[:, 2], c='magenta', label="prediction")
    #
    ax.plot3D(beacon_pos[0, :], beacon_pos[1, :], beacon_pos[2, :], "black", linestyle="dotted", label="reference")

    for ant_nr, ant in enumerate(antenna_list):
        ant_pos = ant.get_antenna_positions()
        ax.scatter3D(ant_pos[0, :], ant_pos[1, :], ant_pos[2, :], label=f"antenna {ant_nr}", s=5)

    plt.legend()

    # plot 2D cross section of trajectory
    plot_2d(pos_real=beacon_pos[:3, :].T, pos_calc=xs[:, :3])

    # compute ecdf
    plt.figure()
    error_vect = utils.error_vector(xs[:, :3].T, beacon_pos)
    res = scipy.stats.ecdf(error_vect)
    ax = plt.subplot()
    res.cdf.plot(ax)
    ax.set_xlabel('Distance Error (m)')
    ax.set_ylabel('Cumulative Error Function')

    # print rmse
    print("RMSE: ", utils.rmse(xs[:, :3].T, beacon_pos))

    # save trajectory data
    save_directory = "test_results/simulation_error_vector"

    Path(save_directory).mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    filename = now.strftime("%m-%d-%Y_%H-%M-%S") + ".npy"
    filepath = os.path.join(save_directory, filename)
    np.save(filepath, error_vect)

    # save phase differences
    # recorded_phi_differences = np.asarray(recorded_phi_differences)
    # recorded_phi_differences = recorded_phi_differences.squeeze()
    # filename = "multipath"
    # np.save(filename, recorded_phi_differences)
    #
    # baseleine_phi_differences = np.asarray(baseleine_phi_differences)
    # baseleine_phi_differences = baseleine_phi_differences.squeeze()
    # filename = "los"
    # np.save(filename, baseleine_phi_differences)

    plt.show()



if __name__ == "__main__":
    np.random.seed(10)
    main(params=Parameters)

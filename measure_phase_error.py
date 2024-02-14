from matplotlib import pyplot as plt
from matplotlib import cm
from antenna_array import AntennaArray
from settings.antenna_element_positions import generate_antenna_element_positions
from settings.config import Parameters as params
import utils
import numpy as np
from beamformer.beamformer import generate_beamformer
import measurement_simulation as sim
import spatial_filter

#################### params
visualize = False
SAMPLE = 1

################### params
np.random.seed(10)

spatial_filter_collection = spatial_filter.SpatialFilter(params=params)

antenna_element_positions, A_full = generate_antenna_element_positions(kind="square_8_8", lmb=params.lmb,
                                                                       get_A_full=True)
antenna_element_positions[[0, 1], :] = antenna_element_positions[[1, 0], :]  # switch x and y rows

beacon_pos = utils.generate_spiral_path(a=1, theta_extent=20, alpha=np.pi / 45)

ant1 = AntennaArray(rot=[0, 0, 45], t=[-2, -3, 3], element_positions=antenna_element_positions)
antenna_list = [ant1]

fs = 100 * params.f
t = np.arange(params.N) / fs

beamformer = generate_beamformer(beamformer_type="delay_and_sum")

avg_filter_error = 0
avg_no_filter_error = 0
for i in range(SAMPLE):
    recorded_phi_differences = []
    recorded_phi_no_multipath_differences = []
    recorded_phi_filtered_differences = []
    for k in range(len(beacon_pos[0])):
        phi_B = np.random.rand() * 2 * np.pi  # transmitter phase at time k

        antenna = antenna_list[0]
        ant_pos = antenna.get_antenna_positions()

        target_dir = beacon_pos[:3, k].reshape((-1, 1)) - antenna.get_t()
        target_dir_r, target_dir_theta, target_dir_phi = utils.cartesian_to_spherical(target_dir[0], target_dir[1],
                                                                                    target_dir[2])

        # multipath_sources = [
        #     {"x": beacon_pos[0, k],
        #      "y": beacon_pos[1, k],
        #      "z": -beacon_pos[2, k],
        #      "a": 0.9 + np.random.randn(1)*0.1}]
        multipath_sources = sim.generate_multipath_sources(params.multipath_count)

        print(f"Target direction theta: {target_dir_theta}, phi: {target_dir_phi}")
        for multipath in multipath_sources:
            dir = np.array([multipath["x"], multipath["y"], multipath["z"]]).reshape((-1, 1)) - antenna.get_t()
            m_r, m_t, m_p = utils.cartesian_to_spherical(dir[0], dir[1], dir[2])
            print(f"Multipath direction theta: {m_t}, phi: {m_p}")

        s_m = sim.measure_s_m_multipath(t=t, antenna_positions=ant_pos,
                                        beacon_pos=beacon_pos[:, k].reshape((-1, 1)),
                                        phi_B=phi_B, sigma=params.sigma, multipath_sources=multipath_sources)

        s_m_no_multipath = sim.measure_s_m_multipath(t=t, antenna_positions=ant_pos,
                                                    beacon_pos=beacon_pos[:, k].reshape((-1, 1)),
                                                    phi_B=phi_B, sigma=0, multipath_sources=[])

        phi = sim.measure_phi(s_m=s_m, f_m=params.f, t=t)
        phi_no_multipath = sim.measure_phi(s_m=s_m_no_multipath, f_m=params.f, t=t)
        recorded_phi_differences.append(utils.mod_2pi(A_full @ phi))
        recorded_phi_no_multipath_differences.append(utils.mod_2pi(A_full @ phi_no_multipath))

        if visualize:

            z = np.exp(1j * ( 2 * np.pi * params.f * (t.reshape((1, -1))) + np.pi)).reshape(-1)
            plt.plot(z.real)

            results, output_signals, thetas, phis = beamformer.compute_beampattern(x=s_m,
                                                                                N_theta=params.N_theta,
                                                                                N_phi=params.N_phi,
                                                                                fs=params.fs,
                                                                                r=ant_pos)

            element_beampattern, theta_e, phi_e = antenna.get_antenna_element_beampattern(thetas=thetas,
                                                                                        phis=phis)
            results *= element_beampattern
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.set_xlim(params.room_x)
            ax.set_ylim(params.room_y)
            ax.set_zlim(params.room_z)

            ax.set_xlabel("x(m)")
            ax.set_ylabel("y(m)")
            ax.set_zlabel("z(m)")

            beampattern_cartesian = beamformer.spherical_to_cartesian(results, thetas=thetas, phis=phis)
            beampattern_cartesian = beampattern_cartesian + antenna.get_t()  # place the pattern on antenna position

            ax.scatter3D(beampattern_cartesian[0, :], beampattern_cartesian[1, :], beampattern_cartesian[2, :])
            ax.plot3D(beacon_pos[0, :], beacon_pos[1, :], beacon_pos[2, :], "green")
            ax.scatter3D(beacon_pos[0, k], beacon_pos[1, k], beacon_pos[2, k], c="red")

            for multipath in multipath_sources:
                ax.scatter3D(multipath["x"], multipath["y"], multipath["z"], "red")

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            theta, phi = np.meshgrid(thetas, phis)

            surf = ax.plot_surface(theta, phi, results, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
            ax.set_xlabel("theta (rad)")
            ax.set_ylabel("phi (rad)")
            ax.set_zlabel("power")
            # Add a color bar which maps values to colors.
            fig.colorbar(surf, shrink=0.5, aspect=5)

        s_m_filtered = s_m
        s_m_filtered = spatial_filter_collection.iterative_max_2D_filter(x=s_m,
                                                              r=ant_pos,
                                                              beamformer=beamformer,
                                                              antenna=antenna,
                                                              peak_threshold=0.1,
                                                              target_theta=target_dir_theta,
                                                              target_phi=target_dir_phi,
                                                              cone_angle=np.deg2rad(params.cone_angle),
                                                              max_iteration=3)

        # s_m_filtered, _ = spatial_filter_collection.two_step_filter(x=s_m,
        #                                             r=ant_pos,
        #                                             beamformer=beamformer,
        #                                             antenna=antenna,
        #                                             peak_threshold=0.1,
        #                                             target_theta=target_dir_theta,
        #                                             target_phi=target_dir_phi,
        #                                             cone_angle=np.deg2rad(params.cone_angle),
        #                                             num_of_removed_signals=1,
        #                                             # target_position=beacon_pos[:3, k].reshape(-1) + np.random.randn(3)*0.05
        #                                             )

        # s_m_filtered = spatial_filter_collection.multipath_filter(x=s_m,
        #                                                       r=ant_pos,
        #                                                       beamformer=beamformer,
        #                                                       antenna=antenna,
        #                                                       peak_threshold=0.3,
        #                                                       target_theta=target_dir_theta,
        #                                                       target_phi=target_dir_phi,
        #                                                       cone_angle=np.deg2rad(params.cone_angle),
        #                                                       )

        # s_m_filtered = spatial_filter_collection.ground_reflection_filter(x=s_m,
        #                                                        r=ant_pos,
        #                                                        beamformer=beamformer,
        #                                                        antenna=antenna,
        #                                                        peak_threshold=0.1,
        #                                                        target_x=beacon_pos[0, k],
        #                                                        target_y=beacon_pos[1, k],
        #                                                        target_z=beacon_pos[2, k],
        #                                                        cone_angle=np.deg2rad(params.cone_angle),
        #                                                        # reflection_position=np.array([multipath["x"], multipath["y"], multipath["z"]])
        #                                                        )

        phi_filtered = sim.measure_phi(s_m=s_m_filtered, f_m=params.f, t=t)
        recorded_phi_filtered_differences.append(utils.mod_2pi(A_full @ phi_filtered))

        if visualize:
            beamformer = generate_beamformer(beamformer_type="delay_and_sum")
            results, output_signals, thetas, phis = beamformer.compute_beampattern(x=s_m_filtered,
                                                                                N_theta=params.N_theta,
                                                                                N_phi=params.N_phi,
                                                                                fs=fs,
                                                                                r=ant_pos)
            results *= element_beampattern
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            ax.set_xlim(params.room_x)
            ax.set_ylim(params.room_y)
            ax.set_zlim(params.room_z)

            ax.set_xlabel("x(m)")
            ax.set_ylabel("y(m)")
            ax.set_zlabel("z(m)")

            beampattern_cartesian = beamformer.spherical_to_cartesian(results, thetas=thetas, phis=phis)
            beampattern_cartesian = beampattern_cartesian + antenna.get_t()  # place the pattern on antenna position

            ax.scatter3D(beampattern_cartesian[0, :], beampattern_cartesian[1, :], beampattern_cartesian[2, :])
            ax.plot3D(beacon_pos[0, :], beacon_pos[1, :], beacon_pos[2, :], "green")
            ax.scatter3D(beacon_pos[0, k], beacon_pos[1, k], beacon_pos[2, k], c="red")

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            theta, phi = np.meshgrid(thetas, phis)

            surf = ax.plot_surface(theta, phi, results, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
            ax.set_xlabel("theta (rad)")
            ax.set_ylabel("phi (rad)")
            ax.set_zlabel("power")
            # Add a color bar which maps values to colors.
            fig.colorbar(surf, shrink=0.5, aspect=5)

            plt.figure()
            for i in range(len(s_m)):
                r = int(np.ceil(np.sqrt(len(s_m))))
                plt.subplot(r, r, i + 1)
                plt.plot((s_m[i, :]), label="s_m")
                plt.plot((s_m_filtered[i, :]), label="s_m filtered")
                plt.plot((s_m_no_multipath[i, :]), label="s_m no multipath")
                plt.legend()

            plt.show()

    recorded_phi_differences = np.asarray(recorded_phi_differences)
    recorded_phi_differences = recorded_phi_differences.squeeze()
    recorded_phi_no_multipath_differences = np.asarray(recorded_phi_no_multipath_differences)
    recorded_phi_no_multipath_differences = recorded_phi_no_multipath_differences.squeeze()
    recorded_phi_filtered_differences = np.asarray(recorded_phi_filtered_differences)
    recorded_phi_filtered_differences = recorded_phi_filtered_differences.squeeze()

    no_filter_error = utils.phase_error(recorded_phi_differences, recorded_phi_no_multipath_differences)
    filter_error = utils.phase_error(recorded_phi_filtered_differences, recorded_phi_no_multipath_differences)

    avg_filter_error += filter_error
    avg_no_filter_error += no_filter_error

print(f"No Filter RMSE: {avg_no_filter_error/SAMPLE}, Filter RMSE: {avg_filter_error/SAMPLE}")

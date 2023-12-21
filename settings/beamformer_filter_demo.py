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

antenna_element_positions, A_full = generate_antenna_element_positions(kind="square_4_4", lmb=params.lmb,
                                                                       get_A_full=True)
antenna_element_positions[[0, 1], :] = antenna_element_positions[[1, 0], :]  # switch x and y rows

beacon_pos = utils.generate_spiral_path(a=1, theta_extent=20, alpha=np.pi / 45)

ant1 = AntennaArray(rot=[0, 45, 45], t=[-1, -1.5, 3], element_positions=antenna_element_positions)
antenna_list = [ant1]

fs = 100 * params.f
t = np.arange(params.N) / fs

beamformer = generate_beamformer(beamformer_type="delay_and_sum")

for k in range(len(beacon_pos[0])):
    phi_B = np.random.rand() * 2 * np.pi  # transmitter phase at time k

    for antenna in antenna_list:
        ant_pos = antenna.get_antenna_positions()
        ant_pos_m_i = ant_pos

        s_m = sim.measure_s_m(t=t, antenna_positions=ant_pos_m_i,
                              beacon_pos=beacon_pos[:, k].reshape((-1, 1)),
                              phi_B=phi_B, sigma=0.01)

        results, output_signals, thetas, phis = beamformer.compute_beampattern(x=s_m,
                                                                               N_theta=params.N_theta,
                                                                               N_phi=params.N_phi,
                                                                               fs=fs,
                                                                               r=ant_pos_m_i)

        if params.apply_element_pattern:
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

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        theta, phi = np.meshgrid(thetas, phis)

        surf = ax.plot_surface(theta, phi, results, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.set_xlabel("theta (rad)")
        ax.set_ylabel("phi (rad)")
        ax.set_zlabel("power")
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)







        s_m_filtered = spatial_filter.remove_max_2D(x=s_m, r=ant_pos_m_i,
                                           results=results, phis=phis, thetas=thetas,
                                           output_signals=output_signals)
        results, output_signals, thetas, phis = beamformer.compute_beampattern(x=s_m_filtered,
                                                                               N_theta=params.N_theta,
                                                                               N_phi=params.N_phi,
                                                                               fs=fs,
                                                                               r=ant_pos_m_i)

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
            r = int(np.sqrt(len(s_m)))
            plt.subplot(r, r, i+1)
            plt.plot(s_m[i, :], label="s_m")
            plt.plot(s_m_filtered[i, :], label="s_m filtered")
            plt.legend()

        plt.show()

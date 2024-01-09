import numpy as np
import scipy.linalg
import spatial_filter
import utils
from antenna_array import AntennaArray
from beamformer.beamformer import generate_beamformer
from settings.antenna_element_positions import generate_antenna_element_positions
from jacobian import Jacobian_h, jacobian_numpy
import matplotlib.pyplot as plt
from settings.config import Parameters as params
from matplotlib import cm
import measurement_simulation as sim

np.random.seed(1)

antenna_element_positions, A_full = generate_antenna_element_positions(kind=params.antenna_kind, lmb=params.lmb, get_A_full=True)
antenna_element_positions[[0, 1], :] = antenna_element_positions[[1, 0], :]  # switch x and y rows

beacon_pos = utils.generate_spiral_path(a=1, theta_extent=20, alpha=np.pi / 45)

ant1 = AntennaArray(rot=[0, 45, 45], t=[-2, -3, 3], element_positions=antenna_element_positions)
ant2 = AntennaArray(rot=[0, 45, 190], t=[4, 1, 3], element_positions=antenna_element_positions)
ant3 = AntennaArray(rot=[0, 45, -60], t=[-2, 3, 3], element_positions=antenna_element_positions)
antenna_list = [ant1, ant2, ant3]

# initial state
xs = []
x = np.array([[beacon_pos[0, 0], beacon_pos[1, 0], beacon_pos[2, 0], 0, 0, 0]]).T
sigma = np.eye(len(x))  # try individual values
jacobian_cache = {}
fs = 100 * params.f
t = np.arange(params.N) / fs
beamformer = generate_beamformer(beamformer_type=params.beamformer_type)
recorded_phi_differences = []


for k in range(len(beacon_pos[0])):
    # prediction
    G = sim.compute_G(params.dt)
    Q = params.sigma_a ** 2 * G @ G.T
    F = sim.compute_F(params.dt)
    x = F @ x
    sigma = F @ sigma @ F.T + Q
    phi_B = np.random.rand() * 2 * np.pi  # transmitter phase at time k
    # multipath_sources = sim.generate_multipath_sources(multipath_count=params.multipath_count)
    multipath_sources = [
        {"x": beacon_pos[0, k],
         "y": beacon_pos[1, k],
         "z": -beacon_pos[2, k],
         "a": 0.9}]
    # iteration
    x_0 = x
    for i in params.i_list:
        ant_pos_i = []
        A = []
        R = []
        h = []
        phi = []

        # to visualize beam pattern at each step
        cartesian_beampattern_list = []
        beampattern_2d_list = []

        for antenna in antenna_list:
            ant_pos = antenna.get_antenna_positions()
            ant_pos_m_i = ant_pos[:, : i]
            ant_pos_i.append(ant_pos_m_i)

            # find target dir with respect to antenna
            target_dir = beacon_pos[:3, k].reshape((-1, 1)) - antenna.get_t()
            target_dir_r, target_dir_theta, target_dir_phi = utils.cartesian_to_spherical(target_dir[0], target_dir[1], target_dir[2])

            for multipath in multipath_sources:
                dir = np.array([multipath["x"], multipath["y"], multipath["z"]]).reshape((-1, 1)) - antenna.get_t()
                m_r, m_t, m_p = utils.cartesian_to_spherical(dir[0], dir[1], dir[2])
                print(f"Multipath direction theta: {m_t}, phi: {m_p}")

            if params.use_multipath:

                if params.measure_phi_m_directly:
                    phi_m, multipath_sources_at_k = sim.measure_multipath(ant_pos_m_i, beacon_pos[:, k].reshape((-1, 1)),
                                                                      sigma_phi=params.sigma_phi,
                                                                      multipath_count=params.multipath_count)
                else:
                    s_m = sim.measure_s_m_multipath(t=t, antenna_positions=ant_pos_m_i,
                                                beacon_pos=beacon_pos[:, k].reshape((-1, 1)),
                                                phi_B=phi_B, sigma=params.sigma, multipath_sources=multipath_sources)

                    if params.apply_element_pattern or params.visualize_beampatterns or params.apply_spatial_filter:
                        results, output_signals, thetas, phis = beamformer.compute_beampattern(x=s_m,
                                                                                               N_theta=params.N_theta,
                                                                                               N_phi=params.N_phi,
                                                                                               fs=fs,
                                                                                               r=ant_pos_m_i)

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

                    if params.apply_spatial_filter:
                        # s_m = spatial_filter.remove_components_2D(x=s_m, r=ant_pos_m_i,
                        #                            results=results, phis=phis, thetas=thetas,
                        #                            output_signals=output_signals)
                        s_m = spatial_filter.iterative_max_2D_filter(x=s_m,
                                                     r=ant_pos_m_i,
                                                     beamformer=beamformer,
                                                     antenna=antenna,
                                                     target_theta=target_dir_theta,
                                                     target_phi=target_dir_phi,
                                                     d_theta=np.deg2rad(params.target_theta_range_deg),
                                                     d_phi=np.deg2rad(params.target_phi_range_deg),
                                                     peak_threshold=0.1,
                                                     max_iteration=1)

                        if params.visualize_beampatterns:
                            results, output_signals, thetas, phis = beamformer.compute_beampattern(x=s_m,
                                                                                                   N_theta=params.N_theta,
                                                                                                   N_phi=params.N_phi,
                                                                                                   fs=fs,
                                                                                                   r=ant_pos_m_i)
                            # results = np.sqrt(results) # power to amplitude conversion
                            beampattern_2d_list[-1].update({"results_filtered": results,
                                                            "thetas_filtered": thetas,
                                                            "phis_filtered": phis,
                                                            "ant_pos_filtered": ant_pos_m_i,
                                                            "s_m_filtered": s_m,
                                                            "output_signals_filtered": output_signals})

                    phi_m = sim.measure_phi(s_m=s_m, f_m=params.f, t=t)


            else:
                if params.measure_phi_m_directly:
                    phi_m = sim.measure(ant_pos_m_i, beacon_pos[:, k].reshape((-1, 1)), sigma_phi=params.sigma_phi)
                else:
                    s_m = sim.measure_s_m(t=t, antenna_positions=ant_pos_m_i, beacon_pos=beacon_pos[:, k].reshape((-1, 1)),
                                      phi_B=phi_B, sigma=params.sigma)
                    phi_m = sim.measure_phi(s_m=s_m, f_m=params.f, t=t)

                if params.visualize_beampatterns:
                    results, output_signals, thetas, phis = beamformer.compute_beampattern(x=s_m,
                                                                                           N_theta=params.N_theta,
                                                                                           N_phi=params.N_phi,
                                                                                           fs=fs,
                                                                                           r=ant_pos_m_i)
                    # results = np.sqrt(results) # power to amplitude conversion
                    beampattern_2d_list.append({"results": results,
                                                "thetas": thetas,
                                                "phis": phis,
                                                "ant_pos": ant_pos_m_i,
                                                "s_m": s_m,
                                                "output_signals": output_signals})
                    beampattern_cartesian = beamformer.spherical_to_cartesian(results, thetas=thetas, phis=phis)
                    beampattern_cartesian = beampattern_cartesian + antenna.get_t()  # place the pattern on antenna position
                    cartesian_beampattern_list.append(beampattern_cartesian)

            phi.append(phi_m)

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
        A = scipy.linalg.block_diag(*A)
        R = scipy.linalg.block_diag(*R)
        h = np.vstack(h)

        z = A @ phi
        h = utils.mod_2pi(A @ h)
        recorded_phi_differences.append(utils.mod_2pi(A @ phi))

        if params.jacobian_type == "scipy":
            if i not in jacobian_cache:
                # use scipy implementation of jacobian (slow)
                h_jacobian = Jacobian_h(N=A.shape[0], I=A.shape[1], w0=2 * np.pi * params.f, c=params.c)
                jacobian_cache[i] = h_jacobian
                h_jacobian.compute_jacobian()
                H = h_jacobian.evaluate_jacobian(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0], ant_pos=ant_pos_i)
            else:
                H = jacobian_cache[i].evaluate_jacobian(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0], ant_pos=ant_pos_i)
        elif params.jacobian_type == "numpy":
            H = jacobian_numpy(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0], ant_pos=ant_pos_i, c=params.c,
                               w0=2 * np.pi * params.f)
        else:
            raise ValueError('jacobian_type is incorrect.')

        K = sigma @ H.T @ np.linalg.inv(R + H @ sigma @ H.T)

        x = x + K @ (utils.mod_2pi(z - h) - H @ (
                x_0 - x))  ################################################## in paper: x_0 - x

    # update
    sigma = (np.eye(len(x)) - K @ H) @ sigma

    xs.append(x)

xs = np.array(xs).squeeze()
# for k, (x, multipath_sources_at_k) in enumerate(zip(xs, multipath_sources)):
#     print("Time step: ", k)
#     print(f"x: {x[0]}, y: {x[1]}, z: {x[2]}, vx: {x[3]}, vy: {x[4]}, vz: {x[5]}")
#     for i, multipath_source in enumerate(multipath_sources_at_k):
#         print(f"Source {i}: ", multipath_source)
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
ax.scatter3D(xs[:, 0], xs[:, 1], xs[:, 2], c='magenta')
#
ax.plot3D(beacon_pos[0, :], beacon_pos[1, :], beacon_pos[2, :], "green")

for ant in antenna_list:
    ant_pos = ant.get_antenna_positions()
    ax.scatter3D(ant_pos[0, :], ant_pos[1, :], ant_pos[2, :])


print("RMSE: ", utils.rmse(xs[:, :3].T, beacon_pos))


# recorded_phi_differences = np.asarray(recorded_phi_differences)
# recorded_phi_differences = recorded_phi_differences.squeeze()
# filename = "multipath" if params.use_multipath else "los"
# np.save(filename, recorded_phi_differences)

plt.show()

import numpy as np
import scipy.linalg
from tqdm import tqdm
import spatial_filter
import utils
from antenna_array import SimulationAntennaArray, RealDataAntennaArray
from beamformer.beamformer import generate_beamformer
from settings.antenna_element_positions import generate_antenna_element_positions
from jacobian import Jacobian_h, jacobian_numpy
import matplotlib.pyplot as plt
from settings.real_data_config import Parameters
from matplotlib import cm
import measurement_simulation as sim
from settings.config import VERBOSE


class DataLoader:
    def __init__(self, folder, params):
        self.params = params
        ant_pos_id10 = np.load(f"{folder}/Antennas_Pos_est_id10.npy").T
        ant_pos_id11 = np.load(f"{folder}/Antennas_Pos_est_id11.npy").T
        ant_pos_id12 = np.load(f"{folder}/Antennas_Pos_est_id12.npy").T
        self.ant_pos_list = [ant_pos_id10, ant_pos_id11, ant_pos_id12]

        phase_offset_id10 = np.load(f"{folder}/Phase_offsets_est_id10.npy")
        phase_offset_id11 = np.load(f"{folder}/Phase_offsets_est_id11.npy")
        phase_offset_id12 = np.load(f"{folder}/Phase_offsets_est_id12.npy")
        self.phase_offset = np.stack([phase_offset_id10, phase_offset_id11, phase_offset_id12], axis=0)

        del ant_pos_id10, ant_pos_id11, ant_pos_id12, phase_offset_id10, phase_offset_id11, phase_offset_id12

        tmp = np.load(f"{folder}/21_no_modulation.npz")
        self.raw_data = tmp["raw_data"]
        self.trajectory_optitrack = tmp["trajectory_optitrack"]
        RECORDING_LENTH = tmp["RECORDING_LENTH"]
        REDUCTION_FACTOR = tmp["REDUCTION_FACTOR"]
        self.nr_of_meas_points = self.trajectory_optitrack.shape[0]
        self.sample_rate = 3.125e6 / REDUCTION_FACTOR
        self.x_0 = np.array(
            [self.trajectory_optitrack[0, 0], self.trajectory_optitrack[0, 1], self.trajectory_optitrack[0, 2]])

    def get_data(self):
        t_list = list(np.linspace(0, self.raw_data.shape[1] * (1 / self.sample_rate), num=self.nr_of_meas_points))
        trajectory_list = list(np.array(
            [self.trajectory_optitrack[:, 0], self.trajectory_optitrack[:, 1], self.trajectory_optitrack[:, 2]]).T)
        raw_data_list = np.array_split(self.raw_data[:, :, :], self.nr_of_meas_points, axis=1)
        return t_list, trajectory_list, raw_data_list

    def get_initial_beacon_pos(self):
        return self.x_0

    def get_antenna_position_list(self):
        return self.ant_pos_list

    def get_A_full(self):
        _, A_full = generate_antenna_element_positions(kind=self.params.antenna_kind, lmb=self.params.lmb, get_A_full=True)
        return A_full

    def get_phase_offset(self):
        return self.phase_offset

def simulate(params):
    spatial_filter_collection = spatial_filter.SpatialFilter(params=params)
    dataloader = DataLoader(folder="./fuer_arda", params=params)

    A_full = dataloader.get_A_full()
    antenna_pos_list = dataloader.get_antenna_position_list()
    antenna_list = []
    for antenna_pos in antenna_pos_list:
        antenna = RealDataAntennaArray(params=params, antenna_element_pos=antenna_pos.T)
        antenna_list.append(antenna)

    # initial state
    xs = []
    initial_beacon_pos = dataloader.get_initial_beacon_pos()
    x = np.array([[initial_beacon_pos[0], initial_beacon_pos[1], initial_beacon_pos[2], 0, 0, 0]]).T

    sigma = np.array([[params.sigma_x0, 0, 0, 0, 0, 0],
                       [0, params.sigma_x0, 0, 0, 0, 0],
                       [0, 0, params.sigma_x0, 0, 0, 0],
                       [0, 0, 0, params.sigma_v0, 0, 0],
                       [0, 0, 0, 0, params.sigma_v0, 0],
                       [0, 0, 0, 0, 0, params.sigma_v0]])
    beamformer = generate_beamformer(beamformer_type=params.beamformer_type)
    recorded_phi_differences = []
    baseleine_phi_differences = []

    t_list, trajectory_list, raw_data_list = dataloader.get_data()
    for k, (t, beacon_pos, raw_data_snip) in tqdm(enumerate(zip(t_list, trajectory_list, raw_data_list))):
        # prediction
        G = sim.compute_G(params.dt)
        Q = params.sigma_a ** 2 * G @ G.T
        F = sim.compute_F(params.dt)
        x = F @ x
        sigma = F @ sigma @ F.T + Q
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

            for antenna_idx, antenna in enumerate(antenna_list):
                ant_pos = antenna.get_antenna_positions()
                ant_pos_m_i = ant_pos[:, : i]
                ant_pos_i.append(ant_pos_m_i)

                # estimate target dir with respect to antenna from estimated position x_0
                target_dir = x_0[:3].reshape((-1, 1)) - antenna.get_t()
                target_dir_r, target_dir_theta, target_dir_phi = utils.cartesian_to_spherical(target_dir[0],
                                                                                              target_dir[1],
                                                                                              target_dir[2])

                real_target_dir = x[:3].reshape((-1, 1)) - antenna.get_t()
                real_target_dir_r, real_target_dir_theta, real_target_dir_phi = utils.cartesian_to_spherical(
                    real_target_dir[0], real_target_dir[1],
                    real_target_dir[2])
                if VERBOSE:
                    print(f"Target direction theta: {real_target_dir_theta}, phi: {real_target_dir_phi}")

                # antennas_used_in_beamformer = params.i_list[0]
                antennas_used_in_beamformer = i

                s_m = raw_data_snip[antenna_idx].T

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
                    # s_m = spatial_filter_collection.remove_components_2D(x=s_m, r=ant_pos,
                    #                            results=results, phis=phis, thetas=thetas,
                    #                            output_signals=output_signals)
                    # s_m = spatial_filter_collection.iterative_max_2D_filter(x=s_m,
                    #                                                       r=ant_pos,
                    #                                                       beamformer=beamformer,
                    #                                                       antenna=antenna,
                    #                                                       peak_threshold=0.1,
                    #                                                       target_theta=target_dir_theta,
                    #                                                       target_phi=target_dir_phi,
                    #                                                       cone_angle=np.deg2rad(
                    #                                                           params.cone_angle),
                    #                                                       max_iteration=params.multipath_count) # needs to be adaptive
                    s_m, _ = spatial_filter_collection.two_step_filter(x=s_m,
                                                                       r=ant_pos,
                                                                       beamformer=beamformer,
                                                                       antenna=antenna,
                                                                       peak_threshold=0.1,
                                                                       target_theta=target_dir_theta,
                                                                       target_phi=target_dir_phi,
                                                                       cone_angle=np.deg2rad(
                                                                           params.cone_angle),
                                                                       num_of_removed_signals=1,
                                                                       antennas_used_in_beamformer=antennas_used_in_beamformer,
                                                                       # uses only the first iteration's antennas in beamformer
                                                                       # target_position=x_0[:3].reshape(-1)
                                                                       )

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

                phi_m = sim.measure_phi(s_m, phase_offset=dataloader.get_phase_offset()[antenna_idx].reshape((-1, 1)))

                phi.append(phi_m[: i])

                phi_m_ref = sim.measure(ant_pos_m_i, beacon_pos.reshape((-1, 1)), sigma_phi=0)
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
                ax.plot3D(beacon_pos[0], beacon_pos[1], beacon_pos[2], "green")
                # ax.scatter3D(trajectory[0, :], beacon_pos[1], beacon_pos[2], c="red")

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

            H = jacobian_numpy(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0], ant_pos=ant_pos_i, c=params.c,
                               w0=2 * np.pi * params.f)


            K = sigma @ H.T @ np.linalg.pinv(R + H @ sigma @ H.T)
            res = (utils.mod_2pi(z - h) - H @ (
                    x_0 - x))
            x = x + K @ res ################################################## in paper: x_0 - x

        # update
        sigma = (np.eye(len(x)) - K @ H) @ sigma
        # print("Beacon position: \n", trajectory_list[k])
        # print("x: \n", x)
        # print("\n\n")
        xs.append(x)

    xs = np.array(xs).squeeze()
    return xs, trajectory_list, antenna_list, recorded_phi_differences, baseleine_phi_differences


def main(params):
    xs, trajectory_list, antenna_list, recorded_phi_differences, baseleine_phi_differences = simulate(params)
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
    ax.plot3D(xs[:, 0], xs[:, 1], xs[:, 2], c='magenta')
    #
    trajectory = np.array(trajectory_list).T
    ax.plot3D(trajectory[0, :], trajectory[1, :], trajectory[2, :], "green")

    for ant in antenna_list:
        ant_pos = ant.get_antenna_positions()
        ax.scatter3D(ant_pos[0, :], ant_pos[1, :], ant_pos[2, :])

    print("RMSE: ", utils.rmse(xs[:, :3].T, trajectory))

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

import numpy as np
import scipy.linalg
from antenna_element_positions import generate_antenna_element_positions
from jacobian import Jacobian_h, jacobian_numpy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from beamformer.beamformer import Beamformer
from config import Parameters as params

# options
jacobian_type = "numpy"  # "numpy" or "scipy"
use_multipath = True  # True or False
antenna_kind = "square_4_4"  # "original" or "square_4_4" or "irregular_4_4"
# options

np.random.seed(1)


def mod_2pi(x):
    mod = np.mod(x, 2 * np.pi)
    mod[mod >= np.pi] -= 2 * np.pi
    return mod


def measure_multipath(antenna_positions, beacon_pos, sigma_phi, multipath_count):
    phi_mix = np.random.rand() * 2 * np.pi
    tau = np.linalg.norm(antenna_positions - beacon_pos, axis=0) / params.c
    phi = -2 * np.pi * params.f * tau + phi_mix
    s_mix = np.exp(1j * phi)
    multipath_sources = []
    for p in range(multipath_count):
        x = np.random.uniform(params.room_x[0], params.room_x[1])
        y = np.random.uniform(params.room_y[0], params.room_y[1])
        z = np.random.uniform(params.room_z[0], params.room_z[1])
        a = np.random.uniform(0, 1)
        multipath_sources.append({"position": [x, y, z],
                                  "amplitude": a})
        multipath_source_pos = np.array([[x, y, z]]).T
        tau = np.linalg.norm(antenna_positions - multipath_source_pos, axis=0) / params.c
        phi = -2 * np.pi * params.f * tau + phi_mix
        s_mix += a * np.exp(1j * phi)

    total_phi = np.angle(s_mix)
    n = np.random.randn(*total_phi.shape) * sigma_phi
    total_phi = mod_2pi(total_phi + n).reshape((-1, 1))
    return total_phi, multipath_sources


def measure(antenna_positions, beacon_pos, sigma_phi):
    # random phase offset for receiver
    phi_mix = np.random.rand() * 2 * np.pi
    # phi_mix = 0
    tau = np.linalg.norm(antenna_positions - beacon_pos, axis=0) / params.c

    n = np.random.randn(*tau.shape) * sigma_phi
    phi = mod_2pi(-2 * np.pi * params.f * tau + phi_mix + n).reshape((-1, 1))
    return phi


def measure_s_m(t, antenna_positions, beacon_pos, phi_B, sigma):
    x = 0
    tau = np.linalg.norm(antenna_positions - beacon_pos, axis=0) / params.c
    phi = 2 * np.pi * params.f * (t.reshape((1, -1)) - tau.reshape((-1, 1))) + phi_B
    x = np.exp(1j * phi)
    N = len(tau)
    n = np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)
    return x + sigma * n


def measure_s_m_multipath(t, antenna_positions, beacon_pos, phi_B, sigma, multipath_sources):
    tau = np.linalg.norm(antenna_positions - beacon_pos, axis=0) / params.c
    phi = 2 * np.pi * params.f * (t.reshape((1, -1)) - tau.reshape((-1, 1))) + phi_B
    s_m = np.exp(1j * phi)

    for p, source in enumerate(multipath_sources):
        x = source["x"]
        y = source["y"]
        z = source["z"]
        a = source["a"]
        multipath_source_pos = np.array([[x, y, z]]).T
        tau = np.linalg.norm(antenna_positions - multipath_source_pos, axis=0) / params.c
        phi = 2 * np.pi * params.f * (t.reshape((1, -1)) - tau.reshape((-1, 1))) + phi_B
        s_m += a * np.exp(1j * phi)
        print(f"Multipath {p}: a:{a}, x:{x}, y:{y}, z:{z}")

    n = np.random.randn(*s_m.shape) + 1j * np.random.randn(*s_m.shape)
    return s_m + sigma * n


def generate_multipath_sources(multipath_count):
    sources = []
    for p in range(multipath_count):
        x = np.random.uniform(params.room_x[0], params.room_x[1])
        y = np.random.uniform(params.room_y[0], params.room_y[1])
        z = np.random.uniform(params.room_z[0], params.room_z[1])
        a = np.random.uniform(0, 1)
        sources.append({"a": a,
                        "x": x,
                        "y": y,
                        "z": z})

    return sources


def compute_G(dt):
    G = np.array(
        [
            [dt ** 2 / 2, 0, 0],
            [0, dt ** 2 / 2, 0],
            [0, 0, dt ** 2 / 2],
            [dt, 0, 0],
            [0, dt, 0],
            [0, 0, dt],
        ]
    )

    return G


def compute_F(dt):
    F = np.array(
        [
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, dt, 0, 0],
            [0, 0, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, dt],
        ]
    )

    return F


def get_A_full(antenna_pos):
    i_max = len(antenna_pos[0]) - 1
    A_full = np.zeros((i_max, len(antenna_pos[0])))

    for i in range(i_max):
        A_full[i, i] = 1
        A_full[i, i + 1] = -1

    return A_full


def generate_spiral_path(a, theta_extent, alpha):
    theta = np.linspace(0, theta_extent)
    scaling = np.linspace(0.5, 1)

    x = a * np.cos(theta) * scaling
    y = a * np.sin(theta) * scaling
    z = a * theta * np.tan(alpha)
    return np.array([x, y, z]).reshape((3, -1))


antenna_element_positions = generate_antenna_element_positions(kind=antenna_kind, lmb=params.lmb)

beacon_pos = generate_spiral_path(a=1, theta_extent=20, alpha=np.pi / 45)

r1 = R.from_euler('xyz', [-45, 0, -30], degrees=True)
R1 = r1.as_matrix()
t1 = np.array([[-1, -1.5, 3]]).T
ant1_pos = R1 @ antenna_element_positions + t1

r2 = R.from_euler('xyz', [-50, 0, 90], degrees=True)
R2 = r2.as_matrix()
t2 = np.array([[1.5, 0, 3]]).T
ant2_pos = R2 @ antenna_element_positions + t2

r3 = R.from_euler('xyz', [-45, 0, -160], degrees=True)
R3 = r3.as_matrix()
t3 = np.array([[-0.5, 1.5, 3]]).T
ant3_pos = R3 @ antenna_element_positions + t3

antenna_transform_list = [{"R": R1, "t": t1}, {"R": R2, "t": t2}, {"R": R3, "t": t3}]
antenna_position_list = [ant1_pos, ant2_pos, ant3_pos]

# initial state
xs = []
x = np.array([[beacon_pos[0, 0], beacon_pos[1, 0], beacon_pos[2, 0], 0, 0, 0]]).T
sigma = np.eye(len(x))  # try individual values
A_full = get_A_full(antenna_element_positions)
jacobian_cache = {}
fs = 100 * params.f
t = np.arange(params.N) / fs
beamformer = Beamformer(type="gpu")

for k in range(len(beacon_pos[0])):
    # prediction
    G = compute_G(params.dt)
    Q = params.sigma_a ** 2 * G @ G.T
    F = compute_F(params.dt)
    x = F @ x
    sigma = F @ sigma @ F.T + Q
    phi_B = np.random.rand() * 2 * np.pi  # transmitter phase at time k
    multipath_sources = generate_multipath_sources(multipath_count=params.multipath_count)
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

        for ant_pos, antenna_transform in zip(antenna_position_list, antenna_transform_list):
            ant_pos_m_i = ant_pos[:, : i]
            ant_pos_i.append(ant_pos_m_i)

            if use_multipath:
                phi_m, multipath_sources_at_k = measure_multipath(ant_pos_m_i, beacon_pos[:, k].reshape((-1, 1)),
                                                                  sigma_phi=params.sigma_phi,
                                                                  multipath_count=params.multipath_count)

                s_m = measure_s_m_multipath(t=t, antenna_positions=ant_pos_m_i,
                                            beacon_pos=beacon_pos[:, k].reshape((-1, 1)),
                                            phi_B=phi_B, sigma=0.1, multipath_sources=multipath_sources)
                results, output_signals, thetas, phis = beamformer.compute_beampattern(x=s_m, N_theta=75, N_phi=75,
                                                                                       fs=fs,
                                                                                       r=ant_pos_m_i)

                beampattern_cartesian = beamformer.spherical_to_cartesian(results, thetas=thetas, phis=phis)
                beampattern_cartesian = beampattern_cartesian + antenna_transform[
                    "t"]  # place the pattern on antenna position
                cartesian_beampattern_list.append(beampattern_cartesian)
            else:
                phi_m = measure(ant_pos_m_i, beacon_pos[:, k].reshape((-1, 1)), sigma_phi=params.sigma_phi)
                s_m = measure_s_m(t=t, antenna_positions=ant_pos_m_i, beacon_pos=beacon_pos[:, k].reshape((-1, 1)),
                                  phi_B=phi_B, sigma=0.1)
                results, output_signals, thetas, phis = beamformer.compute_beampattern(x=s_m, N_theta=75, N_phi=75,
                                                                                       fs=fs,
                                                                                       r=ant_pos_m_i)

                beampattern_cartesian = beamformer.spherical_to_cartesian(results, thetas=thetas, phis=phis)
                beampattern_cartesian = beampattern_cartesian + antenna_transform[
                    "t"]  # place the pattern on antenna position
                cartesian_beampattern_list.append(beampattern_cartesian)

            phi.append(phi_m)

            A_m = A_full[: i - 1, : i]
            A.append(A_m.tolist())

            R_m = params.sigma_phi ** 2 * A_m @ A_m.T
            R.append(R_m.tolist())

            h_m = measure(ant_pos_m_i, x[:3].reshape((-1, 1)), sigma_phi=0)
            h.append(h_m)

        ################################### visualize beampatterns
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
        plt.show()
        ################################### visualize beampatterns

        ant_pos_i = np.hstack(ant_pos_i)
        phi = np.vstack(phi)
        A = scipy.linalg.block_diag(*A)
        R = scipy.linalg.block_diag(*R)
        h = np.vstack(h)

        z = A @ phi
        h = mod_2pi(A @ h)

        if jacobian_type == "scipy":
            if i not in jacobian_cache:
                # use scipy implementation of jacobian (slow)
                h_jacobian = Jacobian_h(N=A.shape[0], I=A.shape[1], w0=2 * np.pi * params.f, c=params.c)
                jacobian_cache[i] = h_jacobian
                h_jacobian.compute_jacobian()
                H = h_jacobian.evaluate_jacobian(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0], ant_pos=ant_pos_i)
            else:
                H = jacobian_cache[i].evaluate_jacobian(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0], ant_pos=ant_pos_i)
        elif jacobian_type == "numpy":
            H = jacobian_numpy(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0], ant_pos=ant_pos_i, c=params.c,
                               w0=2 * np.pi * params.f)
        else:
            raise ValueError('jacobian_type is incorrect.')

        K = sigma @ H.T @ np.linalg.inv(R + H @ sigma @ H.T)

        x = x + K @ (mod_2pi(z - h) - H @ (
                x_0 - x))  ################################################## in paper: x_0 - x

    # update
    sigma = (np.eye(len(x)) - K @ H) @ sigma

    xs.append(x)

# xs = np.array(xs).squeeze()
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
ax.scatter3D(ant1_pos[0, :], ant1_pos[1, :], ant1_pos[2, :], c="red")
ax.scatter3D(ant2_pos[0, :], ant2_pos[1, :], ant2_pos[2, :], c="blue")
ax.scatter3D(ant3_pos[0, :], ant3_pos[1, :], ant3_pos[2, :], c="cyan")
ax.scatter3D(0, 0, 0, c="black")

plt.show()

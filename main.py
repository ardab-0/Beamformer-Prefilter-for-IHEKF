import numpy as np
import scipy.linalg
from sympy_demo import Jacobian_h, jacobian_numpy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

f = 24e9
c = 2.998e8
lmb = c / f
room_x = [-1.5, 1.5]
room_y = [-1.5, 1.5]
room_z = [0, 3]

# options
jacobian_type = "numpy"  # "numpy" or "scipy"
use_multipath = True  # True or False


def mod_2pi(x):
    mod = np.mod(x, 2 * np.pi)
    mod[mod >= np.pi] -= 2 * np.pi
    return mod


def measure_multipath(antenna_positions, beacon_pos, sigma_phi, multipath_count):
    phi_mix = np.random.rand() * 2 * np.pi
    tau = np.linalg.norm(antenna_positions - beacon_pos, axis=0) / c
    phi = -2 * np.pi * f * tau + phi_mix
    s_mix = np.exp(1j * phi)

    for p in range(multipath_count):
        x = np.random.uniform(room_x[0], room_x[1])
        y = np.random.uniform(room_y[0], room_y[1])
        z = np.random.uniform(room_z[0], room_z[1])
        a = np.random.uniform(0, 0.5)
        multipath_source_pos = np.array([[x, y, z]]).T
        tau = np.linalg.norm(antenna_positions - multipath_source_pos, axis=0) / c
        phi = -2 * np.pi * f * tau + phi_mix
        s_mix += a * np.exp(1j * phi)

    total_phi = np.angle(s_mix)
    n = np.random.randn(*total_phi.shape) * sigma_phi
    total_phi = mod_2pi(total_phi + n).reshape((-1, 1))
    return total_phi


def measure(antenna_positions, beacon_pos, sigma_phi):
    # random phase offset for receiver
    phi_mix = np.random.rand() * 2 * np.pi
    # phi_mix = 0
    tau = np.linalg.norm(antenna_positions - beacon_pos, axis=0) / c

    n = np.random.randn(*tau.shape) * sigma_phi
    phi = mod_2pi(-2 * np.pi * f * tau + phi_mix + n).reshape((-1, 1))
    return phi


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


antenna_element_positions = np.array(
    [[0, 0, 0], [2 * lmb, 0, 0], [0, 0, -2 * lmb], [5 * lmb, 0, 0], [0, 0, -5 * lmb], [10 * lmb, 0, 0],
     [0, 0, -10 * lmb], [10 * lmb, 0, -10 * lmb], [-10 * lmb, 0, 0], [-8 * lmb, 0, -5 * lmb]]).T

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

antenna_position_list = [ant1_pos, ant2_pos, ant3_pos]

# initial state
xs = []
x = np.array([[beacon_pos[0, 0], beacon_pos[1, 0], beacon_pos[2, 0], 0, 0, 0]]).T
dt = 0.1
sigma = np.eye(len(x))
sigma_phi = 0.01
sigma_a = 0.1
A_full = get_A_full(antenna_element_positions)
i_list = [3, 5]
jacobian_cache = {}

for k in range(len(beacon_pos[0])):
    # prediction
    G = compute_G(dt)
    Q = sigma_a ** 2 * G @ G.T
    F = compute_F(dt)
    x = F @ x
    sigma = F @ sigma @ F.T + Q

    # iteration
    x_0 = x
    for i in i_list:
        ant_pos_i = []
        A = []
        R = []
        h = []
        phi = []
        for ant_pos in antenna_position_list:
            ant_pos_m_i = ant_pos[:, : i]
            ant_pos_i.append(ant_pos_m_i)

            if use_multipath:
                phi_m = measure_multipath(ant_pos_m_i, beacon_pos[:, k].reshape((-1, 1)), sigma_phi=sigma_phi,
                                          multipath_count=4)
            else:
                phi_m = measure(ant_pos_m_i, beacon_pos[:, k].reshape((-1, 1)), sigma_phi=sigma_phi)
            phi.append(phi_m)

            A_m = A_full[: i - 1, : i]
            A.append(A_m.tolist())

            R_m = sigma_phi ** 2 * A_m @ A_m.T
            R.append(R_m.tolist())

            h_m = measure(ant_pos_m_i, x[:3].reshape((-1, 1)), sigma_phi=0)
            h.append(h_m)

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
                h_jacobian = Jacobian_h(N=A.shape[0], I=A.shape[1], w0=2 * np.pi * f, c=c)
                jacobian_cache[i] = h_jacobian
                h_jacobian.compute_jacobian()
                H = h_jacobian.evaluate_jacobian(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0], ant_pos=ant_pos_i)
            else:
                H = jacobian_cache[i].evaluate_jacobian(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0], ant_pos=ant_pos_i)
        elif jacobian_type == "numpy":
            H = jacobian_numpy(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0], ant_pos=ant_pos_i, c=c, w0=2 * np.pi * f)
        else:
            raise ValueError('jacobian_type is incorrect.')

        K = sigma @ H.T @ np.linalg.inv(R + H @ sigma @ H.T)

        x = x + K @ (mod_2pi(z - h) - H @ (
                x_0 - x))  ################################################## in paper: x_0 - x

    # update
    sigma = (np.eye(len(x)) - K @ H) @ sigma

    xs.append(x)

xs = np.array(xs).squeeze()
for x in xs:
    print(f"x: {x[0]}, y: {x[1]}, z: {x[2]}, vx: {x[3]}, vy: {x[4]}, vz: {x[5]}")
# creating an empty canvas
fig = plt.figure()

ax = plt.axes(projection="3d")
ax.set_xlim(room_x)
ax.set_ylim(room_y)
ax.set_zlim(room_z)

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

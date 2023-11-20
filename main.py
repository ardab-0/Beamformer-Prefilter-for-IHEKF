import numpy as np
import scipy.linalg

from sympy_demo import Jacobian_h, jacobian_numpy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R

f = 24e9
c = 2.998e8
lmb = c / f


def mod_2pi(x):
    mod = np.mod(x, 2 * np.pi)
    mod[mod >= np.pi] -= 2 * np.pi
    return mod


def measure(antenna_positions, beacon_pos):
    # random phase offset for receiver
    phi_mix = np.random.rand() * 2 * np.pi

    tau = np.linalg.norm(antenna_positions - beacon_pos, axis=0) / c
    phi = mod_2pi(-2 * np.pi * f * tau + phi_mix).reshape((-1, 1))
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
        A_full[i, 0] = 1
        A_full[i, i + 1] = -1

    return A_full


ant_pos = np.array([[0, 0, 0], [2 * lmb, 0, 0], [0, 0, -2 * lmb], [5 * lmb, 0, 0], [0, 0, -5 * lmb], [10 * lmb, 0, 0],
                    [0, 0, -10 * lmb], [10 * lmb, 0, -10 * lmb], [-10 * lmb, 0, 0], [-8 * lmb, 0, -5 * lmb]]).T
i_list = [3, 3]

beacon_pos = np.array(
    [
        [0, 0, 0],
        [0, 0, 0.05],
        [0, 0, 0.1],
        [0, 0, 0.15],
        [0, 0, 0.2],
        [0, 0, 0.25],
        [0, 0, 0.30],
        [0, 0, 0.35],
        [0, 0, 0.4],
        [0, 0, 0.45],
        [0, 0, 0.50],
        [0, 0, 0.55],
        [0, 0, 0.6],
        [0, 0, 0.65],
        [0, 0, 0.70],
        [0, 0, 0.75],
        [0, 0, 0.80],
        [0, 0, 0.85],
        [0, 0, 0.90],
        [0, 0, 0.95],
        [0, 0, 1],
        [0, 0.1, 1],
        [0, 0.2, 1],
        [0, 0.3, 1],
        [0, 0.4, 1],
        [0, 0.5, 1],
        [0, 0.6, 1],
        [0, 0.7, 1],
        [0, 0.8, 1],
        [0, 0.9, 1],
        [0, 1, 1],
        [0, 1, 0.9],
        [0, 1, 0.8],
        [0, 1, 0.7],
        [0, 1, 0.6],
        [0, 1, 0.5]
    ]
).T

r1 = R.from_euler('xyz', [-45, 0, 0], degrees=True)
R1 = r1.as_matrix()
t1 = np.array([[0, -1, 2]]).T
ant1_pos = R1 @ ant_pos + t1

r2 = R.from_euler('xyz', [-45, 0, 90], degrees=True)
R2 = r2.as_matrix()
t2 = np.array([[1, 0, 2]]).T
ant2_pos = R2 @ ant_pos + t2

r3 = R.from_euler('xyz', [-45, 0, -135], degrees=True)
R3 = r3.as_matrix()
t3 = np.array([[-1, 1, 2]]).T
ant3_pos = R3 @ ant_pos + t3

print(measure(ant_pos, beacon_pos[:, 0].reshape((-1, 1))))

antenna_position_list = [ant1_pos, ant2_pos, ant3_pos]

# initial state
xs = []
x = np.array([[0, 0, 0, 0, 0, 0]]).T
dt = 0.1
sigma = np.eye(len(x))*10
sigma_phi = 0.00001
sigma_a = 0.9
A_full = get_A_full(ant_pos)
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

            phi_m = measure(ant_pos_m_i, beacon_pos[:, k].reshape((-1, 1)))
            phi.append(phi_m)

            A_m = A_full[: i - 1, : i]
            A.append(A_m.tolist())

            R_m = sigma_phi ** 2 * A_m @ A_m.T
            R.append(R_m.tolist())

            h_m = measure(ant_pos_m_i, x[:3].reshape((-1, 1)))
            h.append(h_m)

        ant_pos_i = np.hstack(ant_pos_i)
        phi = np.vstack(phi)
        A = scipy.linalg.block_diag(*A)
        R = scipy.linalg.block_diag(*R)
        h = np.vstack(h)

        z = A @ phi
        h = mod_2pi(A @ h)

        H = jacobian_numpy(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0], ant_pos=ant_pos_i, c=c, w0=2 * np.pi * f)
        # if i not in jacobian_cache:
        #
        #     # use scipy implementation of jacobian (slow)
        #     # h_jacobian = Jacobian_h(N=A.shape[0], I=A.shape[1], w0=2 * np.pi * f, c=c)
        #     # jacobian_cache[i] = h_jacobian
        #     #
        #     # h_jacobian.compute_jacobian()
        #     # H = h_jacobian.evaluate_jacobian(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0], ant_pos=ant_pos_i)
        #
        #     # use numpy implementation of jacobian
        #     H = jacobian_numpy(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0], ant_pos=ant_pos_i, c=c, w0=2 * np.pi * f)
        #     print("a")
        # else:
        #     H = jacobian_cache[i].evaluate_jacobian(A_np=A, px=x[0, 0], py=x[1, 0], pz=x[2, 0], ant_pos=ant_pos_i)

        K = sigma @ H.T @ np.linalg.inv(R + H @ sigma @ H.T)

        x = x + K @ (mod_2pi(z - h) - H @ (x_0 - x))

    # update
    sigma = (np.eye(len(x)) - K @ H) @ sigma

    xs.append(x)

xs = np.array(xs).squeeze()
print(xs)

# creating an empty canvas
fig = plt.figure()

ax = plt.axes(projection="3d")
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])

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

plt.show()

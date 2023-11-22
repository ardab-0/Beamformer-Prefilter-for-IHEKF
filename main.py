import numpy as np
import scipy.linalg
import json
from jacobian import Jacobian_h, jacobian_numpy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R
import streamlit as st
import plotly.graph_objects as go


f = 24e9
c = 2.998e8
lmb = c / f


def mod_2pi(x):
    mod = np.mod(x, 2 * np.pi)
    mod[mod >= np.pi] -= 2 * np.pi
    return mod


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


def antenna_data(filename):
    with open(filename) as user_file:
        file_contents = user_file.read()
    parsed_json = json.loads(file_contents)

    return parsed_json


ant_pos = np.array([[0, 0, 0], [2 * lmb, 0, 0], [0, 0, -2 * lmb], [5 * lmb, 0, 0], [0, 0, -5 * lmb], [10 * lmb, 0, 0],
                    [0, 0, -10 * lmb], [10 * lmb, 0, -10 * lmb], [-10 * lmb, 0, 0], [-8 * lmb, 0, -5 * lmb]]).T



class Antenna:
    def __init__(self, antenna_element_pos: np.ndarray, rot_deg: list, ant_pos_m: list, idx: int):
        """

        :param antenna_element_pos: 3xN
        :param rot_deg: degree [xrot, yrot, zrot]
        :param ant_pos_m: meter [x, y, z]
        :param idx:antenna id
        """
        self._ant_pos_m = ant_pos_m
        self._rot_deg = rot_deg
        self._antenna_element_pos = antenna_element_pos
        self._idx = idx

    def get_R(self):
        r1 = R.from_euler('xyz', self._ant_pos_m, degrees=True)
        R1 = r1.as_matrix()
        return R1

    def get_t(self):
        t1 = np.array([self._ant_pos_m]).T
        return t1

    def get_antenna_positions(self):
        R1 = self.get_R()
        t1 = self.get_t()
        return R1 @ self._antenna_element_pos + t1

    def get_idx(self):
        return self._idx


beacon_pos = generate_spiral_path(a=1, theta_extent=20, alpha=np.pi / 45)

if 'antenna_data' not in st.session_state:
    st.session_state.antenna_data = antenna_data(filename="antenna_pos.json")


antenna_idx_to_adjust = st.sidebar.selectbox(
    'Select antenna idx to adjust',
    range(len(st.session_state.antenna_data)) )


with st.sidebar.form('accounting'):
    col1, col2 = st.columns(2)
    with col1:
        xrot = st.number_input('X Rotation (Deg)', value=st.session_state.antenna_data[antenna_idx_to_adjust]["rot_deg"][0], step=1)
        yrot = st.number_input('Y Rotation (Deg)', value=st.session_state.antenna_data[antenna_idx_to_adjust]["rot_deg"][1], step=1)
        zrot = st.number_input('Z Rotation (Deg)', value=st.session_state.antenna_data[antenna_idx_to_adjust]["rot_deg"][2], step=1)
    with col2:
        x = st.number_input('X (m)', value=float(st.session_state.antenna_data[antenna_idx_to_adjust]["ant_pos_m"][0]), step=0.1)
        y = st.number_input('Y (m)', value=float(st.session_state.antenna_data[antenna_idx_to_adjust]["ant_pos_m"][1]), step=0.1)
        z = st.number_input('Z (m)', value=float(st.session_state.antenna_data[antenna_idx_to_adjust]["ant_pos_m"][2]), step=0.1)
    # add other input here.

    submit = st.form_submit_button('save')

if submit:
    st.session_state.antenna_data[antenna_idx_to_adjust]["rot_deg"][0] = xrot
    st.session_state.antenna_data[antenna_idx_to_adjust]["rot_deg"][1] = yrot
    st.session_state.antenna_data[antenna_idx_to_adjust]["rot_deg"][2] = zrot

    st.session_state.antenna_data[antenna_idx_to_adjust]["ant_pos_m"][0] = x
    st.session_state.antenna_data[antenna_idx_to_adjust]["ant_pos_m"][1] = y
    st.session_state.antenna_data[antenna_idx_to_adjust]["ant_pos_m"][2] = z

antennas = []
for i, antenna_pos in enumerate(st.session_state.antenna_data):
    antennas.append(Antenna(antenna_element_pos=ant_pos, rot_deg=antenna_pos["rot_deg"], ant_pos_m=antenna_pos["ant_pos_m"], idx=i))

antenna_position_list = []
for antenna in antennas:
    antenna_position_list.append(antenna.get_antenna_positions())

# initial state
xs = []
x = np.array([[beacon_pos[0, 0], beacon_pos[1, 0], beacon_pos[2, 0], 0, 0, 0]]).T
dt = 0.1
sigma = np.eye(len(x))
sigma_phi = 0.01
sigma_a = 0.15
A_full = get_A_full(ant_pos)
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

    # measure all antennas
    phi_measurements = []
    for ant_pos in antenna_position_list:
        phi_measurements.append(measure(ant_pos, beacon_pos[:, k].reshape((-1, 1)), sigma_phi=sigma_phi))

    for i in i_list:
        ant_pos_i = []
        A = []
        R = []
        h = []
        phi = []
        for j, ant_pos in enumerate(antenna_position_list):
            ant_pos_m_i = ant_pos[:, : i]
            ant_pos_i.append(ant_pos_m_i)

            phi_m = phi_measurements[j][:i]
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

        x = x + K @ (mod_2pi(z - h) - H @ (
                x_0 - x))  ################################################## in paper: x_0 - x

    # update
    sigma = (np.eye(len(x)) - K @ H) @ sigma

    xs.append(x)

xs = np.array(xs).squeeze()
print(xs)

# # creating an empty canvas
# fig = plt.figure()
#
# ax = plt.axes(projection="3d")
# ax.set_xlim([-1.5, 1.5])
# ax.set_ylim([-1.5, 1.5])
# ax.set_zlim([-0.5, 2.5])
#
# ax.set_xlabel("x(m)")
# ax.set_ylabel("y(m)")
# ax.set_zlabel("z(m)")
#
# # ax.plot3D(xs[:, 0], xs[:, 1], xs[:, 2], 'red')
# ax.scatter3D(xs[:, 0], xs[:, 1], xs[:, 2], c='magenta')
# #
# ax.plot3D(beacon_pos[0, :], beacon_pos[1, :], beacon_pos[2, :], "green")
fig = go.Figure()

for i, ant_pos in enumerate(antenna_position_list):
    # ax.scatter3D(ant_pos[0, :], ant_pos[1, :], ant_pos[2, :], label=str(i))

    scatter = fig.add_trace(go.Scatter3d(
        x=ant_pos[0, :],
        y=ant_pos[1, :],
        z=ant_pos[2, :],
        mode='markers',
        marker=dict(

            colorscale='Viridis',  # Choosing a color scale
            size=2
        )

    ))

# ax.scatter3D(0, 0, 0, c="black")





fig.update_layout(
    scene=dict(
        xaxis=dict(title='X Axis', range=[-1.5, 1.5]),
        yaxis=dict(title='Y Axis', range=[-1.5, 1.5]),
        zaxis=dict(title='Z Axis', range=[-0.5, 3.5])
    )
)

st.plotly_chart(fig,  use_container_width=True)

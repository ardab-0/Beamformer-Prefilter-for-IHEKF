import numpy as np
import measurement_simulation as sim


def generate_antenna_element_positions(kind: str, lmb: float, get_A_full: bool = False) -> np.ndarray | tuple[
    np.ndarray, np.ndarray]:
    if kind == "original":
        A_full = np.array([[1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [1, 0, -1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, -1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, -1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, -1, 0, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, -1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0, -1, 0, 0],
                           [1, 0, 0, 0, 0, 0, 0, 0, -1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, -1]])
        positions = np.array(
            [[0, 0, 0], [2 * lmb, 0, 0], [0, 0, -2 * lmb], [5 * lmb, 0, 0], [0, 0, -5 * lmb], [10 * lmb, 0, 0],
             [0, 0, -10 * lmb], [10 * lmb, 0, -10 * lmb], [-10 * lmb, 0, 0], [-8 * lmb, 0, -5 * lmb]]).T
        if get_A_full:
            return positions, A_full
        else:
            return positions

    elif kind == "square_4_4":

        antenna_element_positions = np.array(
            [[0, 0, 0], [lmb / 2, 0, 0], [lmb, 0, 0], [3 * lmb / 2, 0, 0],
             [0, 0, -lmb / 2], [lmb / 2, 0, -lmb / 2], [2 * lmb / 2, 0, -lmb / 2], [3 * lmb / 2, 0, -lmb / 2],
             [0, 0, -2 * lmb / 2], [lmb / 2, 0, -2 * lmb / 2], [2 * lmb / 2, 0, -2 * lmb / 2],
             [3 * lmb / 2, 0, -2 * lmb / 2],
             [0, 0, -3 * lmb / 2], [lmb / 2, 0, -3 * lmb / 2], [2 * lmb / 2, 0, -3 * lmb / 2],
             [3 * lmb / 2, 0, -3 * lmb / 2]]).T

        A_full = sim.get_A_full(antenna_element_positions)

        return antenna_element_positions, A_full

    elif kind == "square_4_3":
        d = 1
        antenna_element_positions = np.array(
            [[0, 0, 0 * lmb * d], [1 * lmb * d, 0, 0 * lmb * d], [2 * lmb * d, 0, 0 * lmb * d],
             [0, 0, 1 * lmb * d], [1 * lmb * d, 0, 1 * lmb * d], [2 * lmb * d, 0, 1 * lmb * d],
             [0, 0, 2 * lmb * d], [1 * lmb * d, 0, 2 * lmb * d], [2 * lmb * d, 0, 2 * lmb * d],
             [0, 0, 3 * lmb * d], [1 * lmb * d, 0, 3 * lmb * d], [2 * lmb * d, 0, 3 * lmb * d],
             # [0, 0, 4 * lmb * d], [1 * lmb * d, 0, 4 * lmb * d], [2 * lmb * d, 0, 4 * lmb * d],
             # [0, 0, 5 * lmb * d], [1 * lmb * d, 0, 5 * lmb * d], [2 * lmb * d, 0, 5 * lmb * d],
             # [0, 0, 6 * lmb * d], [1 * lmb * d, 0, 6 * lmb * d], [2 * lmb * d, 0, 6 * lmb * d],
             # [0, 0, 7 * lmb * d], [1 * lmb * d, 0, 7 * lmb * d], [2 * lmb * d, 0, 7 * lmb * d],
             ]).T

        A_full = sim.get_A_full(antenna_element_positions)

        return antenna_element_positions, A_full
    elif kind == "irregular_4_4":
        antenna_element_positions = np.array(
            [[0, 0, 0], [lmb / 2, 0, 0], [lmb, 0, 0], [3 * lmb / 2, 0, 0],
             [0, 0, -lmb / 2], [lmb / 2, 0, -lmb / 2], [2 * lmb / 2, 0, -lmb / 2], [3 * lmb / 2, 0, -lmb / 2],
             [0, 0, -2 * lmb / 2], [lmb / 2, 0, -2 * lmb / 2], [2 * lmb / 2, 0, -2 * lmb / 2],
             [3 * lmb / 2, 0, -2 * lmb / 2],
             [0, 0, -3 * lmb / 2], [12 * lmb, 0, -3 * lmb / 2], [24 * lmb / 2, 0, -3 * lmb / 2],
             [48 * lmb / 2, 0, -3 * lmb / 2]]).T
        A_full = sim.get_A_full(antenna_element_positions)

        return antenna_element_positions, A_full

    elif kind == "regular_8_2":
        antenna_element_positions = np.array(
            [[0, 0, 0], [lmb / 2, 0, 0],
             [0, 0, lmb / 2], [lmb / 2, 0, lmb / 2],
             [0, 0, 2 * lmb / 2], [lmb / 2, 0, 2 * lmb / 2],
             [0, 0, 3 * lmb / 2], [lmb / 2, 0, 3 * lmb / 2],
             [0, 0, 4 * lmb / 2], [lmb / 2, 0, 4 * lmb / 2],
             [0, 0, 5 * lmb / 2], [lmb / 2, 0, 5 * lmb / 2],
             [0, 0, 6 * lmb / 2], [lmb / 2, 0, 6 * lmb / 2],
             [0, 0, 7 * lmb / 2], [lmb / 2, 0, 7 * lmb / 2]
             ]).T
        A_full = sim.get_A_full(antenna_element_positions)

        return antenna_element_positions, A_full

    elif kind == "irregular_8_2":
        d = 0.8
        antenna_element_positions = np.array(
            [[0, 0, 0 * lmb * d], [lmb * d, 0, 0 * lmb * d],
             [0, 0, lmb * d], [lmb * d, 0, lmb * d],
             [0, 0, 2 * lmb * d], [lmb * d, 0, 2 * lmb * d],
             [0, 0, 3 * lmb * d], [lmb * d, 0, 3 * lmb * d],
             [0, 0, 4 * lmb * d], [lmb * d, 0, 4 * lmb * d],
             [0, 0, 5 * lmb * d], [lmb * d, 0, 5 * lmb * d],
             [0, 0, 9 * lmb * d], [lmb * d, 0, 9 * lmb * d],

             [0, 0, 13 * lmb * d], [lmb * d, 0, 13 * lmb * d]
             ]).T
        A_full = sim.get_A_full(antenna_element_positions)
        # A_full = np.array([[1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                    [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                    [1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                    [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                    [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                    [1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                    [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        #                    [1, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
        #                    [1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        #                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
        #                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0],
        #                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],
        #                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        #                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        return antenna_element_positions, A_full

    elif kind == "regular_16_1":
        antenna_element_positions = np.array(
            [[0, 0, 0],
             [0, 0, lmb / 2],
             [0, 0, 2 * lmb / 2],
             [0, 0, 3 * lmb / 2],
             [0, 0, 4 * lmb / 2],
             [0, 0, 5 * lmb / 2],
             [0, 0, 6 * lmb / 2],
             [0, 0, 7 * lmb / 2],
             [0, 0, 8 * lmb / 2],
             [0, 0, 9 * lmb / 2],
             [0, 0, 10 * lmb / 2],
             [0, 0, 11 * lmb / 2],
             [0, 0, 12 * lmb / 2],
             [0, 0, 13 * lmb / 2],
             [0, 0, 14 * lmb / 2],
             [0, 0, 15 * lmb / 2]
             ]).T
        A_full = sim.get_A_full(antenna_element_positions)

        return antenna_element_positions, A_full

    elif kind == "irregular_16_1":
        antenna_element_positions = np.array(
            [[0, 0, 0],
             [0, 0, lmb / 2],
             [0, 0, 2 * lmb / 2],
             [0, 0, 3 * lmb / 2],
             [0, 0, 4 * lmb / 2],
             [0, 0, 5 * lmb / 2],
             [0, 0, 6 * lmb / 2],
             [0, 0, 7 * lmb / 2],
             [0, 0, 9 * lmb / 2],
             [0, 0, 11 * lmb / 2],
             [0, 0, 13 * lmb / 2],
             [0, 0, 15 * lmb / 2],
             [0, 0, 17 * lmb / 2],
             [0, 0, 19 * lmb / 2],
             [0, 0, 21 * lmb / 2],
             [0, 0, 23 * lmb / 2]
             ]).T
        A_full = sim.get_A_full(antenna_element_positions)

        return antenna_element_positions, A_full

    elif kind == "plus":
        antenna_element_positions = np.array(
            [[0, 0, 0],
             [0, 0, lmb / 2],
             [0, 0, 2 * lmb / 2],
             [0, 0, 3 * lmb / 2],
             [0, 0, 4 * lmb / 2],
             [0, 0, 5 * lmb / 2],
             [0, 0, 6 * lmb / 2],
             [0, 0, 7 * lmb / 2],
             [lmb / 2, 0, 0],
             [2 * lmb / 2, 0, 0],
             [3 * lmb / 2, 0, 0],
             [4 * lmb / 2, 0, 0],
             [5 * lmb / 2, 0, 0],
             [6 * lmb / 2, 0, 0],
             [7 * lmb / 2, 0, 0],
             [8 * lmb / 2, 0, 0]
             ]).T

        A_full = sim.get_A_full(antenna_element_positions)

        return antenna_element_positions, A_full
    elif kind == "square_6_6":
        t = np.arange(0, 6) * lmb / 2
        x, y = np.meshgrid(t, t)

        antenna_element_positions = np.stack([x, np.zeros_like(x), -y]).reshape((3, -1))

        A_full = sim.get_A_full(antenna_element_positions)

        return antenna_element_positions, A_full

    elif kind == "square_8_8":
        t = np.arange(0, 8) * lmb / 2
        x, y = np.meshgrid(t, t)

        antenna_element_positions = np.stack([x, np.zeros_like(x), -y]).reshape((3, -1))

        A_full = sim.get_A_full(antenna_element_positions)

        return antenna_element_positions, A_full
    elif kind == "square_16_16":
        t = np.arange(0, 16) * lmb / 2
        x, y = np.meshgrid(t, t)

        antenna_element_positions = np.stack([x, np.zeros_like(x), -y]).reshape((3, -1))

        A_full = sim.get_A_full(antenna_element_positions)

        return antenna_element_positions, A_full

    else:
        raise ValueError("Wrong antenna kind.")

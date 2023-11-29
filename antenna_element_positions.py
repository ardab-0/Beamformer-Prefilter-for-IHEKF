import numpy as np


def generate_antenna_element_positions(kind: str, lmb) -> np.array:
    if kind == "original":
        return np.array(
            [[0, 0, 0], [2 * lmb, 0, 0], [0, 0, -2 * lmb], [5 * lmb, 0, 0], [0, 0, -5 * lmb], [10 * lmb, 0, 0],
             [0, 0, -10 * lmb], [10 * lmb, 0, -10 * lmb], [-10 * lmb, 0, 0], [-8 * lmb, 0, -5 * lmb]]).T
    elif kind == "square_4_4":
        return np.array(
            [[0, 0, 0], [lmb / 2, 0, 0], [lmb, 0, 0], [3 * lmb / 2, 0, 0],
             [0, 0, -lmb / 2], [lmb / 2, 0, -lmb / 2], [2 * lmb / 2, 0, -lmb / 2], [3 * lmb / 2, 0, -lmb / 2],
             [0, 0, -2 * lmb / 2], [lmb / 2, 0, -2 * lmb / 2], [2 * lmb / 2, 0, -2 * lmb / 2],
             [3 * lmb / 2, 0, -2 * lmb / 2],
             [0, 0, -lmb / 2], [lmb / 2, 0, -3 * lmb / 2], [2 * lmb / 2, 0, -3 * lmb / 2],
             [3 * lmb / 2, 0, -3 * lmb / 2]]).T
    else:
        raise ValueError("Wrong antenna kind.")

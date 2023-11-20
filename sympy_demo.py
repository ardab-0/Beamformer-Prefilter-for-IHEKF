from sympy import MatrixSymbol, Matrix
import sympy as s
import numpy as np


def jacobian(A_np, px, py, pz, w, c, ant_pos):
    N, I = A_np.shape
    
    x, y, z, w0, vx, vy, vz = s.symbols('x, y, z, w0, vx, vy, vz')

    pB = s.Matrix( [x, y, z ] )
    A = MatrixSymbol('A', N, I)
    phi_mix = MatrixSymbol("phi_mix", I, 1)

    pMs = []

    for i in range(I):
        pM = MatrixSymbol(f"pM^{i}", 3, 1)
        pMs.append(pM)

        tau_row = s.sqrt( (pB - pM).T * (pB - pM) ) / c
        tau_row = s.Matrix(tau_row)
        if i == 0:
            tau = tau_row
        else:
            tau = tau.row_insert(i, tau_row)



    phi = - w0 * tau + phi_mix
    h = A * phi
    h = Matrix(h)

    H = h.jacobian(Matrix([x, y, z, vx, vy, vz]))
  


    subs = {A: s.Matrix(A_np), 
            x: px,
            y: py,
            z: pz,
            w0: w
            }


    for i, pM in enumerate(pMs):
        subs[pM] = s.Matrix(ant_pos[:, i])


    F = np.array(H.evalf(subs=subs)).astype(float)
    return F

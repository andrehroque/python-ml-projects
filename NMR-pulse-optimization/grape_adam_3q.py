# Gradient Ascent - GRAPE - 3 qubits --- detuning --- ADAM

import numpy as np
import os
import json
import re
import ast
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from qiskit.quantum_info import Pauli
from numba import njit

np.set_printoptions(linewidth=200, precision=4, suppress=True)

pauli_x = Pauli('X')
pauli_y = Pauli('Y')
pauli_z = Pauli('Z')
identity = Pauli('I')

X = pauli_x.to_matrix()
Y = pauli_y.to_matrix()
Z = pauli_z.to_matrix()
I = identity.to_matrix()

def kron3(A, B, C):
    return np.kron(A, np.kron(B, C))

Ix1 = kron3(X/2, I, I)
Iy1 = kron3(Y/2, I, I)
Iz1 = kron3(Z/2, I, I)

Ix2 = kron3(I, X/2, I)
Iy2 = kron3(I, Y/2, I)
Iz2 = kron3(I, Z/2, I)

Ix3 = kron3(I, I, X/2)
Iy3 = kron3(I, I, Y/2)
Iz3 = kron3(I, I, Z/2)

U_cnot = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,0,1],
                   [0,0,1,0]], dtype=np.complex128)

U_perm = np.array([[1,0,0,0],
                   [0,0,0,1],
                   [0,1,0,0],
                   [0,0,1,0]], dtype=np.complex128)

U_cz = np.array([[1,0,0,0],
                [0,1,0,0],
                [0,0,1,0],
                [0,0,0,-1]], dtype=np.complex128)

U_toffoli = np.array([[1,0,0,0,0,0,0,0],
                      [0,1,0,0,0,0,0,0],
                      [0,0,1,0,0,0,0,0],
                      [0,0,0,1,0,0,0,0],
                      [0,0,0,0,1,0,0,0],
                      [0,0,0,0,0,1,0,0],
                      [0,0,0,0,0,0,0,1],
                      [0,0,0,0,0,0,1,0]], dtype=np.complex128)

U_fredkin1 = np.array([[1,0,0,0,0,0,0,0],
                      [0,1,0,0,0,0,0,0],
                      [0,0,1,0,0,0,0,0],
                      [0,0,0,1,0,0,0,0],
                      [0,0,0,0,1,0,0,0],
                      [0,0,0,0,0,0,1,0],
                      [0,0,0,0,0,1,0,0],
                      [0,0,0,0,0,0,0,1]], dtype=np.complex128)

U_fredkin2 = np.array([[1,0,0,0,0,0,0,0],
                      [0,1,0,0,0,0,0,0],
                      [0,0,1,0,0,0,0,0],
                      [0,0,0,0,0,0,1,0],
                      [0,0,0,0,1,0,0,0],
                      [0,0,0,0,0,1,0,0],
                      [0,0,0,1,0,0,0,0],
                      [0,0,0,0,0,0,0,1]], dtype=np.complex128)

U_fredkin3 = np.array([[1,0,0,0,0,0,0,0],
                      [0,1,0,0,0,0,0,0],
                      [0,0,1,0,0,0,0,0],
                      [0,0,0,0,0,1,0,0],
                      [0,0,0,0,1,0,0,0],
                      [0,0,0,1,0,0,0,0],
                      [0,0,0,0,0,0,1,0],
                      [0,0,0,0,0,0,0,1]], dtype=np.complex128)

# constants
n = 3
d = 2**n
w1_max = 6447

chem1 = 14.51
chem2 = -918.51
chem3 = 673.62
J12 = -134.28
J13 = 67.14
J23 = 57.98

H_x = 2*np.pi*w1_max*(Ix1+Ix2+Ix3)
H_y = 2*np.pi*w1_max*(Iy1+Iy2+Iy3)
H_z = 2*np.pi* (Iz1+Iz2+Iz3)

ZZI = kron3(Z/2,Z/2,I)
ZIZ = kron3(Z/2,I,Z/2)
IZZ = kron3(I,Z/2,Z/2)

H_J = 2*np.pi* (chem1*Iz1 + chem2*Iz2 + chem3*Iz3) + 2*np.pi* (J12*ZZI + J13*ZIZ + J23*IZZ)

@njit
def fast_expm(H, dt):
    w, v = np.linalg.eigh(H)
    exp_diag = np.exp(-1j * w * dt)
    return v @ np.diag(exp_diag) @ v.conj().T

@njit
def p_calc_3q(Rx, Ry, Rz, U_target, T, N, H_x, H_y, H_z, H_J):
    d = 8
    P_list = np.zeros((N, d, d), dtype=np.complex128)
    P = U_target.copy()
    dt = T / N

    for k in range(N-1, -1, -1):
        H = H_J.copy()
        H -= Rz[k] * H_z
        H += Rx[k] * H_x
        H += Ry[k] * H_y

        Uk = fast_expm(H, dt)
        P = Uk.conj().T @ P
        P_list[k] = P

    return P_list
  
@njit
def x_calc_3q(Rx, Ry, Rz, T, N, H_x, H_y, H_z, H_J):
    d = 8
    X_list = np.zeros((N, d, d), dtype=np.complex128)
    U = np.eye(d, dtype=np.complex128)
    dt = T / N

    for k in range(N):
        H = H_J.copy()
        H -= Rz[k] * H_z
        H += Rx[k] * H_x
        H += Ry[k] * H_y

        Uk = fast_expm(H, dt)
        U = Uk @ U
        X_list[k] = U

    return X_list, U

def fidelity_grape(U, U_target):
    phi = np.trace(U_target.conj().T @ U)
    return np.abs(phi)**2/d**2

@njit
def polar_to_cartesian(Amp, Phi):
    Rx = Amp * np.cos(Phi)
    Ry = Amp * np.sin(Phi)
    return Rx, Ry

@njit
def clip_vector(Rx, Ry, max_norm=1.0):
    norm = np.sqrt(Rx**2 + Ry**2)
    scale = np.maximum(1.0, norm / max_norm)
    return Rx / scale, Ry / scale

@njit
def gaussiansquare_envelope(N, rise_frac=0.15):
    rise_len = int(N * rise_frac)
    flat_len = N - 2 * rise_len

    sigma = rise_len / 3.0
    t = np.arange(rise_len)

    rise = np.exp(-0.5 * ((t - rise_len)**2) / sigma**2)
    fall = rise[::-1]

    flat = np.ones(flat_len)

    env = np.concatenate((rise, flat, fall))

    return env / np.max(env)

@njit
def grape_grad_3q(Rx, Ry, U_target, T, N, H_x, H_y, H_J):
    d = 8
    dt = T / N

    X_list, U_final = x_calc_3q(Rx, Ry, T, N, H_x, H_y, H_J)
    P_list = p_calc_3q(Rx, Ry, U_target, T, N, H_x, H_y, H_J)

    phi = np.trace(U_target.conj().T @ U_final)
    F0 = np.abs(phi)**2 / d**2

    grad_Rx = np.zeros(N)
    grad_Ry = np.zeros(N)

    for k in range(N):
        Xk = X_list[k]
        Pk = P_list[k]

        dX_Rx = -1j * dt * (H_x @ Xk)
        dX_Ry = -1j * dt * (H_y @ Xk)

        grad_Rx[k] = 2 * np.real(
            np.trace(Pk.conj().T @ dX_Rx) * np.conj(phi)
        )

        grad_Ry[k] = 2 * np.real(
            np.trace(Pk.conj().T @ dX_Ry) * np.conj(phi)
        )

    return F0, grad_Rx, grad_Ry, U_final

@njit
def grape_grad_3q_envelope(
    Rx, Ry, Rz, U_target, T, N,
    H_x, H_y, H_z, H_J, env
):
    d = 8
    dt = T / N
    
    Rz_env = env * Rz
    Rx_env = env * Rx
    Ry_env = env * Ry

    X_list, U_final = x_calc_3q(
        Rx_env, Ry_env, Rz_env, T, N, H_x, H_y, H_z, H_J
    )
    P_list = p_calc_3q(
        Rx_env, Ry_env, Rz_env, U_target, T, N, H_x, H_y, H_z, H_J
    )

    phi = np.trace(U_target.conj().T @ U_final)
    F0 = np.abs(phi)**2 / d**2

    grad_Rx = np.zeros(N)
    grad_Ry = np.zeros(N)
    grad_Rz = np.zeros(N)
    
    for k in range(N):
        Xk = X_list[k]
        Pk = P_list[k]

        dX_Rx = -1j * dt * (H_x @ Xk)
        dX_Ry = -1j * dt * (H_y @ Xk)
        dX_Rz = -1j * dt * (-H_z @ Xk)
        
        grad_Rx[k] = 2*np.real(
            np.trace(Pk.conj().T @ dX_Rx) * np.conj(phi)
        )

        grad_Ry[k] = 2*np.real(
            np.trace(Pk.conj().T @ dX_Ry) * np.conj(phi)
        )

        grad_Rz[k] = 2*np.real(
            np.trace(Pk.conj().T @ dX_Rz) * np.conj(phi)
        )
        
        grad_Rx[k] *= env[k]
        grad_Ry[k] *= env[k]
        grad_Rz[k] *= env[k]
        

    return F0, grad_Rx, grad_Ry, grad_Rz, U_final

def make_envelope(N, rise_frac):
    if rise_frac <= 0:
        return np.ones(N)
    return gaussiansquare_envelope(N, rise_frac)


def smooth_cost_and_grad(x, env, lam, N):
    """
    Smoothness penalty on the physical (enveloped) signal: env*x.
    Returns scalar cost and gradient w.r.t. x (not x_phys).
    """
    x_phys = env * x
    diffs  = np.diff(x_phys)
    cost   = lam * np.sum(diffs**2) / N

    g_phys = np.zeros(N)
    g_phys[:-1] -= 2 * diffs
    g_phys[1:]  += 2 * diffs
    g_phys      *= lam / N

    # chain rule: d(cost)/dx[k] = d(cost)/d(x_phys[k]) * env[k]
    return cost, g_phys * env


def adam_grape(
    cost_and_grad_fn, params0, bounds,
    lr=1e-2, beta1=0.9, beta2=0.999,
    eps=1e-8, max_iter=3000, tol=1e-7,
    print_every=200
):
    params    = params0.copy()
    m         = np.zeros_like(params)
    v         = np.zeros_like(params)
    best_cost = np.inf
    best_params = params.copy()

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    for t in range(1, max_iter + 1):
        cost, grad = cost_and_grad_fn(params)

        if cost < best_cost:
            best_cost   = cost
            best_params = params.copy()

        if np.linalg.norm(grad) < tol:
            print(f"  ADAM converged at iteration {t}, cost={cost:.6f}")
            break

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2

        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)

        params = params - lr * m_hat / (np.sqrt(v_hat) + eps)
        params = np.clip(params, lo, hi)

        if print_every and t % print_every == 0:
            print(f"  ADAM iter {t:4d} | cost={cost:.6f} | F={1-cost:.6f} | |grad|={np.linalg.norm(grad):.2e}")

    return best_params, best_cost


def pulse_optimize_grape_adam_lbfgs_3q(
    U_target, T, N,
    startParameters=None,
    adam_lr=5e-3,
    adam_iter=3000,
    lbfgs_iter=1000,
    lambda_smooth=1e-3,
    delta_RF=0.05,
    gaus_rise_frac=0.15,
    det_bound=1000,
    print_every=200
):
    """
    GRAPE optimizer: ADAM exploration followed by L-BFGS-B polishing.
    Cartesian (Rx, Ry, Rz) parameterization throughout.

    Args:
        U_target:        target unitary (8x8)
        T:               total pulse duration (s)
        N:               number of time slices
        startParameters: (Rx, Ry, Rz) each length N, or None for random init
        adam_lr:         ADAM learning rate
        adam_iter:       max ADAM iterations
        lbfgs_iter:      max L-BFGS-B iterations
        lambda_smooth:   smoothness penalty weight
        delta_RF:        RF robustness perturbation fraction
        gaus_rise_frac:  Gaussian square rise fraction (0 = flat/no envelope)
        det_bound:       detuning parameter bound (Hz)
        print_every:     print progress every N ADAM iterations (0 = silent)
    """

    if startParameters is not None:
        Rx0, Ry0, Rz0 = startParameters
    else:
        angle = np.random.uniform(-np.pi, np.pi, N)
        amp   = np.random.uniform(0.05, 0.3, N)
        Rx0   = amp * np.cos(angle)
        Ry0   = amp * np.sin(angle)
        Rz0   = np.zeros(N)

    params0 = np.concatenate([Rx0, Ry0, Rz0])

    bounds = [(-1, 1)] * (2 * N) + [(-det_bound, det_bound)] * N

    env = make_envelope(N, gaus_rise_frac)

    perturbations = [1 - delta_RF, 1.0, 1 + delta_RF]

    def cost_function(params):
        Rx = params[:N]
        Ry = params[N:2*N]
        Rz = params[2*N:]

        F_total       = 0.0
        gRx_total     = np.zeros(N)
        gRy_total     = np.zeros(N)
        gRz_total     = np.zeros(N)

        for scale in perturbations:
            F, gRx, gRy, gRz, _ = grape_grad_3q_envelope(
                Rx * scale, Ry * scale, Rz,
                U_target, T, N,
                H_x, H_y, H_z, H_J, env
            )
            F_total   += F / len(perturbations)
            gRx_total += gRx * scale / len(perturbations)
            gRy_total += gRy * scale / len(perturbations)
            gRz_total += gRz        / len(perturbations)

        sc_x, sg_x = smooth_cost_and_grad(Rx, env, lambda_smooth, N)
        sc_y, sg_y = smooth_cost_and_grad(Ry, env, lambda_smooth, N)
        sc_z, sg_z = smooth_cost_and_grad(Rz, env, lambda_smooth, N)

        cost = 1.0 - F_total + sc_x + sc_y + sc_z

        grad = np.concatenate([
            -gRx_total + sg_x,
            -gRy_total + sg_y,
            -gRz_total + sg_z,
        ])

        return cost, grad

    # ADAM
    params_adam, cost_adam = adam_grape(
        cost_function, params0, bounds,
        lr=adam_lr, max_iter=adam_iter,
        print_every=print_every
    )
    print(f"ADAM done | best cost={cost_adam:.6f} | F≈{1-cost_adam:.6f}")

    # LBFGS to find best minimum around ADAM result
    res = minimize(
        cost_function,
        params_adam,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': lbfgs_iter, 'ftol': 1e-15, 'gtol': 1e-8}
    )
    print(f"LBFGS done | cost={res.fun:.6f} | F≈{1-res.fun:.6f} | iters={res.nit}")

    Rx_opt = res.x[:N]
    Ry_opt = res.x[N:2*N]
    Rz_opt = res.x[2*N:]

    F_final, _, _, _, U_final = grape_grad_3q_envelope(
        Rx_opt, Ry_opt, Rz_opt,
        U_target, T, N,
        H_x, H_y, H_z, H_J, env
    )

    return Rx_opt, Ry_opt, Rz_opt, F_final, U_final
   

    
dt     = 35e-6
slices = np.arange(500, 1001, 10)
Fs     = []

for i, N in enumerate(slices):
    T = dt * N

    Rx, Ry, Rz, F_final, U_final = pulse_optimize_grape_adam_lbfgs_3q(
        U_fredkin1, T, N,
        startParameters=None,
        adam_lr=5e-3,
        adam_iter=3000,
        lbfgs_iter=1000,
        lambda_smooth=1e-3,
        delta_RF=0.05,
        gaus_rise_frac=0.15,
        det_bound=1000,
        print_every=500
    )

    print(f"N={N} | Final fidelity: {F_final:.6f}")
    Fs.append(F_final)

Ts = slices * dt
plt.figure(figsize=(8, 6))
plt.plot(Ts * 1e3, Fs, linewidth=2)
plt.xlabel("Pulse Duration (ms)")
plt.ylabel("Fidelity")
plt.title("GRAPE Fredkin1 | ADAM + L-BFGS-B")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
# Gradient Ascent - GRAPE - different channels

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
identity2 = Pauli('I')

X = pauli_x.to_matrix()
Y = pauli_y.to_matrix()
Z = pauli_z.to_matrix()
I = identity2.to_matrix()

Ix1 = np.kron(X/2,I)
Iy1 = np.kron(Y/2,I)
Iz1 = np.kron(Z/2,I)
Ix2 = np.kron(I,X/2)
Iy2 = np.kron(I,Y/2)
Iz2 = np.kron(I,Z/2)

SINGLE_STATES = {
    "0": np.array([1, 0], dtype=np.complex128),
    "1": np.array([0, 1], dtype=np.complex128),
    "+": np.array([1, 1], dtype=np.complex128) / np.sqrt(2),
    "-": np.array([1, -1], dtype=np.complex128) / np.sqrt(2),
}

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

U_swap =  np.array([[1,0,0,0],
                    [0,0,1,0],
                    [0,1,0,0],
                    [0,0,0,1]], dtype=np.complex128)

Hadamard = 1/np.sqrt(2) * np.array([[1,1],
                                    [1,-1]], dtype=np.complex128)

# constants
n = 2
d = 2**n
#wH = 27e6
#wP = 11e6
#w1H = w1P after calibration
w1_max = 6250
J=696

H_x_H = 2*np.pi*w1_max*(Ix1)
H_x_P = 2*np.pi*w1_max*(Ix2)
H_y_H = 2*np.pi*w1_max*(Iy1)
H_y_P = 2*np.pi*w1_max*(Iy2)
XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZZ = np.kron(Z, Z)
H_J = 2*np.pi*J*(ZZ/4)
# H_J = 2*np.pi*J*((XX+YY+ZZ)/4)

# H_0 = 0 because working on resonance. Only include when detuning or counteracting chemical shift mismatch

@njit
def fast_expm(H, dt):
    w, v = np.linalg.eigh(H)
    exp_diag = np.exp(-1j * w * dt)
    return v @ np.diag(exp_diag) @ v.conj().T

@njit
def p_calc_2channel(Rx_H, Ry_H, Rx_P, Ry_P, U_target, J, T, N,
                    H_x_H, H_y_H, H_x_P, H_y_P, H_J):
    """
    GRAPE algorithm backwards propagation calculation.
    """
    d = 4
    P_list = np.zeros((N, d, d), dtype=np.complex128)
    P = U_target.copy()
    dt = T / N

    for k in range(N-1, -1, -1):
        H = H_J.copy()
        H += Rx_H[k]*H_x_H
        H += Ry_H[k]*H_y_H
        H += Rx_P[k]*H_x_P
        H += Ry_P[k]*H_y_P
        Uk = fast_expm(H, dt)
        P = Uk.conj().T @ P
        P_list[k] = P

    return P_list
  
@njit
def x_calc_2channel(Rx_H, Ry_H, Rx_P, Ry_P, J, T, N,
                    H_x_H, H_y_H, H_x_P, H_y_P, H_J):
    """
    GRAPE algorithm forwards propagation calculation.
    """  
    d = 4
    X_list = np.zeros((N, d, d), dtype=np.complex128)
    U = np.eye(d, dtype=np.complex128)
    dt = T / N

    for k in range(N):
        H = H_J.copy()
        H += Rx_H[k]*H_x_H
        H += Ry_H[k]*H_y_H
        H += Rx_P[k]*H_x_P
        H += Ry_P[k]*H_y_P
        Uk = fast_expm(H, dt)
        U = Uk @ U
        X_list[k] = U

    return X_list, U

def fidelity_grape(U, U_target):
    phi = np.trace(U_target.conj().T @ U)
    return np.abs(phi)**2/d**2

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
def grape_grad_2channel(Rx_H, Ry_H, Rx_P, Ry_P, U_target, J, T, N,
                        H_x_H, H_y_H, H_x_P, H_y_P, H_J):
    """
    GRAPE algorithm gradient calculation.
    """
    d = 4
    dt = T / N
    X_list, U_final = x_calc_2channel(Rx_H, Ry_H, Rx_P, Ry_P, J, T, N,
                                      H_x_H, H_y_H, H_x_P, H_y_P, H_J)
    P_list = p_calc_2channel(Rx_H, Ry_H, Rx_P, Ry_P, U_target, J, T, N,
                             H_x_H, H_y_H, H_x_P, H_y_P, H_J)

    phi = np.trace(U_target.conj().T @ U_final)
    F0 = (np.abs(phi)**2) / (d**2)

    grad_Rx_H = np.zeros(N)
    grad_Ry_H = np.zeros(N)
    grad_Rx_P = np.zeros(N)
    grad_Ry_P = np.zeros(N)

    for k in range(N):
        Xk = X_list[k]
        Pk = P_list[k]

        dX_Rx_H = -1j * dt * (H_x_H @ Xk)
        dX_Ry_H = -1j * dt * (H_y_H @ Xk)
        dX_Rx_P = -1j * dt * (H_x_P @ Xk)
        dX_Ry_P = -1j * dt * (H_y_P @ Xk)

        grad_Rx_H[k] = 2 * np.real(np.trace(Pk.conj().T @ dX_Rx_H) * np.conj(phi))
        grad_Ry_H[k] = 2 * np.real(np.trace(Pk.conj().T @ dX_Ry_H) * np.conj(phi))
        grad_Rx_P[k] = 2 * np.real(np.trace(Pk.conj().T @ dX_Rx_P) * np.conj(phi))
        grad_Ry_P[k] = 2 * np.real(np.trace(Pk.conj().T @ dX_Ry_P) * np.conj(phi))

    return F0, grad_Rx_H, grad_Ry_H, grad_Rx_P, grad_Ry_P, U_final

@njit
def grape_grad_2channel_envelope(Rx_H, Ry_H, Rx_P, Ry_P, U_target, J, T, N,
                        H_x_H, H_y_H, H_x_P, H_y_P, H_J, env):
    """
    GRAPE algorithm gradient calculation.
    """
    d = 4
    dt = T / N
    
    Rx_H_enveloped = env * Rx_H
    Ry_H_enveloped = env * Ry_H
    Rx_P_enveloped = env * Rx_P
    Ry_P_enveloped = env * Ry_P
    
    X_list, U_final = x_calc_2channel(Rx_H_enveloped, Ry_H_enveloped, Rx_P_enveloped, Ry_P_enveloped, J, T, N,
                                      H_x_H, H_y_H, H_x_P, H_y_P, H_J)
    P_list = p_calc_2channel(Rx_H_enveloped, Ry_H_enveloped, Rx_P_enveloped, Ry_P_enveloped, U_target, J, T, N,
                             H_x_H, H_y_H, H_x_P, H_y_P, H_J)

    phi = np.trace(U_target.conj().T @ U_final)
    F0 = (np.abs(phi)**2) / (d**2)

    grad_Rx_H = np.zeros(N)
    grad_Ry_H = np.zeros(N)
    grad_Rx_P = np.zeros(N)
    grad_Ry_P = np.zeros(N)

    for k in range(N):
        Xk = X_list[k]
        Pk = P_list[k]

        dX_Rx_H = -1j * dt * (H_x_H @ Xk)
        dX_Ry_H = -1j * dt * (H_y_H @ Xk)
        dX_Rx_P = -1j * dt * (H_x_P @ Xk)
        dX_Ry_P = -1j * dt * (H_y_P @ Xk)

        grad_Rx_H[k] = 2 * np.real(np.trace(Pk.conj().T @ dX_Rx_H) * np.conj(phi))
        grad_Ry_H[k] = 2 * np.real(np.trace(Pk.conj().T @ dX_Ry_H) * np.conj(phi))
        grad_Rx_P[k] = 2 * np.real(np.trace(Pk.conj().T @ dX_Rx_P) * np.conj(phi))
        grad_Ry_P[k] = 2 * np.real(np.trace(Pk.conj().T @ dX_Ry_P) * np.conj(phi))
        
        grad_Rx_H[k] *= env[k]
        grad_Ry_H[k] *= env[k]
        grad_Rx_P[k] *= env[k]
        grad_Ry_P[k] *= env[k]

    return F0, grad_Rx_H, grad_Ry_H, grad_Rx_P, grad_Ry_P, U_final

def pulse_optimize_grape(U_target, J, T, N, learn_rate, max_iter, startParameters=None):
    """
    Run optimization loop up to max iterations or theoretical convergence: fidelity >= 0.999.
    """
    # Initialize
    if startParameters is not None:
        Rx_H, Ry_H, Rx_P, Ry_P = startParameters
    else:
        Rx_H = np.full(N, 0.2, dtype=np.float64)
        Ry_H = np.full(N, 0.2, dtype=np.float64)
        Rx_P = np.full(N, 0.2, dtype=np.float64)
        Ry_P = np.full(N, 0.2, dtype=np.float64)

    lr = learn_rate
    maxi = max_iter

    for i in range(maxi):
        F, gRxH, gRyH, gRxP, gRyP, U_final = grape_grad_2channel(
            Rx_H, Ry_H, Rx_P, Ry_P,
            U_target, J, T, N,
            H_x_H, H_y_H, H_x_P, H_y_P, H_J
        )

        Rx_H += lr * gRxH
        Ry_H += lr * gRyH
        Rx_P += lr * gRxP
        Ry_P += lr * gRyP

        # Clip amplitudes
        Rx_H, Ry_H = clip_vector(Rx_H, Ry_H)
        Rx_P, Ry_P = clip_vector(Rx_P, Ry_P)

        if F > 0.999:
            print(f"Converged at iteration {i}")
            break

    print("Final fidelity:", F)
    return Rx_H, Ry_H, Rx_P, Ry_P, F

def pulse_optimize_grape_lbfgs(U_target, J, T, N,
                                      H_x_H, H_y_H, H_x_P, H_y_P, H_J,
                                      startParameters=None,
                                      max_iter=1000,
                                      lambda_smooth=1e-3,
                                      delta_RF=0.05):
    """
    Optimize GRAPE pulses using L-BFGS-B with:
        - Smoothness penalty
        - Robustness to RF amplitude fluctuations
    """

    # ---- Initialize pulses ----
    if startParameters is not None:
        Rx_H, Ry_H, Rx_P, Ry_P = startParameters
    else:
        Rx_H = np.full(N, 0.2)
        Ry_H = np.full(N, 0.2)
        Rx_P = np.full(N, 0.2)
        Ry_P = np.full(N, 0.2)

    params0 = np.concatenate([Rx_H, Ry_H, Rx_P, Ry_P])

    bounds = [(-1, 1)]*len(params0)  # amplitude bounds

    # ---- Cost function wrapper ----
    def grape_cost_robust_smooth(params):
        # Unpack pulses
        Rx_H = params[0:N]
        Ry_H = params[N:2*N]
        Rx_P = params[2*N:3*N]
        Ry_P = params[3*N:4*N]

        # ---- Robustness: average over small RF variations ----
        perturbations = [1 - delta_RF, 1, 1 + delta_RF]
        F_total = 0
        grad_Rx_H_total = np.zeros(N)
        grad_Ry_H_total = np.zeros(N)
        grad_Rx_P_total = np.zeros(N)
        grad_Ry_P_total = np.zeros(N)

        for scale in perturbations:
            Rx_H_s = Rx_H * scale
            Ry_H_s = Ry_H * scale
            Rx_P_s = Rx_P * scale
            Ry_P_s = Ry_P * scale

            F, gRxH, gRyH, gRxP, gRyP, _ = grape_grad_2channel(
                Rx_H_s, Ry_H_s, Rx_P_s, Ry_P_s,
                U_target, J, T, N,
                H_x_H, H_y_H, H_x_P, H_y_P, H_J
            )

            F_total += F / len(perturbations)
            grad_Rx_H_total += gRxH / len(perturbations) * scale
            grad_Ry_H_total += gRyH / len(perturbations) * scale
            grad_Rx_P_total += gRxP / len(perturbations) * scale
            grad_Ry_P_total += gRyP / len(perturbations) * scale

        # ---- Smoothness penalty ----
        dRx_H = np.diff(Rx_H, prepend=Rx_H[0])
        dRy_H = np.diff(Ry_H, prepend=Ry_H[0])
        dRx_P = np.diff(Rx_P, prepend=Rx_P[0])
        dRy_P = np.diff(Ry_P, prepend=Ry_P[0])

        smooth_cost = lambda_smooth * (np.sum(dRx_H**2) + np.sum(dRy_H**2)
                                       + np.sum(dRx_P**2) + np.sum(dRy_P**2))

        grad_smooth_Rx_H = 2*lambda_smooth*dRx_H
        grad_smooth_Ry_H = 2*lambda_smooth*dRy_H
        grad_smooth_Rx_P = 2*lambda_smooth*dRx_P
        grad_smooth_Ry_P = 2*lambda_smooth*dRy_P

        # ---- Total cost and gradient ----
        cost = 1 - F_total + smooth_cost
        grad_flat = np.concatenate([
            -grad_Rx_H_total + grad_smooth_Rx_H,
            -grad_Ry_H_total + grad_smooth_Ry_H,
            -grad_Rx_P_total + grad_smooth_Rx_P,
            -grad_Ry_P_total + grad_smooth_Ry_P
        ])

        return cost, grad_flat

    # ---- Run L-BFGS-B optimization ----
    res = minimize(grape_cost_robust_smooth, params0,
                   method='L-BFGS-B', jac=True, bounds=bounds,
                   options={'maxiter': max_iter})

    # ---- Extract pulses ----
    Rx_H_opt = res.x[0:N]
    Ry_H_opt = res.x[N:2*N]
    Rx_P_opt = res.x[2*N:3*N]
    Ry_P_opt = res.x[3*N:4*N]

    # ---- Compute final fidelity & final unitary ----
    F_final, _, _, _, _, U_final = grape_grad_2channel(
        Rx_H_opt, Ry_H_opt, Rx_P_opt, Ry_P_opt,
        U_target, J, T, N,
        H_x_H, H_y_H, H_x_P, H_y_P, H_J
    )

    return Rx_H_opt, Ry_H_opt, Rx_P_opt, Ry_P_opt, F_final, U_final

def pulse_optimize_grape_lbfgs_envelope(U_target, J, T, N,
                                      H_x_H, H_y_H, H_x_P, H_y_P, H_J,
                                      startParameters=None,
                                      max_iter=1000,
                                      lambda_smooth=1e-3,
                                      delta_RF=0.05, gaus_rise_frac=0.15):
    """
    Optimize GRAPE pulses using L-BFGS-B with:
        - Smoothness penalty
        - Robustness to RF amplitude fluctuations
        - Gaussian Square envelope
        
    Args:
        U_target: target unitary matrix of result state to achieve
        J: J-coupling constant
        T: total pulse duration
        N: number of segmentations of pulse duration
        H_x_H, H_y_H, H_x_P, H_y_P: fixed Hamiltionians
        startParameters: initial state if not optimizing pulse from scratch
        max_iter: maximum permitted iterations for scipy.minimize function
        lambda_smooth: smoothing factor
        delta_RF: RF field perturbation factor (percentage)
        gaus_rise_frac: gaussian square rising section fraction of whole pulse duration
    """

    # ---- Initialize pulses ----
    if startParameters is not None:
        Rx_H, Ry_H, Rx_P, Ry_P = startParameters
    else:
        Rx_H = np.full(N, 0.2)
        Ry_H = np.full(N, 0.2)
        Rx_P = np.full(N, 0.2)
        Ry_P = np.full(N, 0.2)

    params0 = np.concatenate([Rx_H, Ry_H, Rx_P, Ry_P])

    bounds = [(-1, 1)]*len(params0)  # amplitude bounds
    
    env = gaussiansquare_envelope(N, gaus_rise_frac)

    # ---- Cost function wrapper ----
    def grape_cost_robust_smooth(params):
        # Unpack pulses
        Rx_H = params[0:N]
        Ry_H = params[N:2*N]
        Rx_P = params[2*N:3*N]
        Ry_P = params[3*N:4*N]
        
        # Clip amplitude vector to make sure that exceeding maximum amplitude of 100 is avoided
        Rx_H, Ry_H = clip_vector(Rx_H, Ry_H)
        Rx_P, Ry_P = clip_vector(Rx_P, Ry_P)

        # ---- Robustness: average over small RF variations ----
        perturbations = [1 - delta_RF, 1, 1 + delta_RF]
        F_total = 0
        
        # Prepare gradients to update
        grad_Rx_H_total = np.zeros(N)
        grad_Ry_H_total = np.zeros(N)
        grad_Rx_P_total = np.zeros(N)
        grad_Ry_P_total = np.zeros(N)

        for scale in perturbations:
            
            # Detuned control values
            Rx_H_scaled = Rx_H * scale
            Ry_H_scaled = Ry_H * scale
            Rx_P_scaled = Rx_P * scale
            Ry_P_scaled = Ry_P * scale
            
            # Calculate gradient with detuned control values
            F, gRxH, gRyH, gRxP, gRyP, _ = grape_grad_2channel_envelope(
                Rx_H_scaled, Ry_H_scaled, Rx_P_scaled, Ry_P_scaled,
                U_target, J, T, N,
                H_x_H, H_y_H, H_x_P, H_y_P, H_J,
                env
            )

            # Coalesce into gradient values to update control on iteration
            F_total += F / len(perturbations)
            grad_Rx_H_total += gRxH / len(perturbations) * scale
            grad_Ry_H_total += gRyH / len(perturbations) * scale
            grad_Rx_P_total += gRxP / len(perturbations) * scale
            grad_Ry_P_total += gRyP / len(perturbations) * scale

        # ---- Smoothness penalty ----
        
        # Smoothness penalties should be calculated on actual physical control values, apply the envelope
        Rx_H_physical = env * Rx_H
        Ry_H_physical = env * Ry_H
        Rx_P_physical = env * Rx_P
        Ry_P_physical = env * Ry_P
        
        # Calculate differences between adjacent control values
        dRx_H = np.diff(Rx_H_physical, prepend=Rx_H_physical[0])
        dRy_H = np.diff(Ry_H_physical, prepend=Ry_H_physical[0])
        dRx_P = np.diff(Rx_P_physical, prepend=Rx_P_physical[0])
        dRy_P = np.diff(Ry_P_physical, prepend=Ry_P_physical[0])

        # If the differences are large, smooth_cost grows larger
        smooth_cost = lambda_smooth * (np.sum(dRx_H**2) + np.sum(dRy_H**2)
                                       + np.sum(dRx_P**2) + np.sum(dRy_P**2))

        grad_smooth_Rx_H = 2*lambda_smooth*dRx_H
        grad_smooth_Ry_H = 2*lambda_smooth*dRy_H
        grad_smooth_Rx_P = 2*lambda_smooth*dRx_P
        grad_smooth_Ry_P = 2*lambda_smooth*dRy_P

        # ---- Total cost and gradient ----
        cost = 1 - F_total + smooth_cost
        grad_flat = np.concatenate([
            -grad_Rx_H_total + grad_smooth_Rx_H,
            -grad_Ry_H_total + grad_smooth_Ry_H,
            -grad_Rx_P_total + grad_smooth_Rx_P,
            -grad_Ry_P_total + grad_smooth_Ry_P
        ])

        return cost, grad_flat

    # ---- Run L-BFGS-B optimization ----
    res = minimize(grape_cost_robust_smooth, params0,
                   method='L-BFGS-B', jac=True, bounds=bounds,
                   options={'maxiter': max_iter})

    # ---- Extract pulses ----
    Rx_H_opt = res.x[0:N]
    Ry_H_opt = res.x[N:2*N]
    Rx_P_opt = res.x[2*N:3*N]
    Ry_P_opt = res.x[3*N:4*N]
    
    # Make sure its physically possible ---> can lead to mismatch between what the l-bfgs optimizer got and actual results
    Rx_H_opt, Ry_H_opt = clip_vector(Rx_H_opt, Ry_H_opt)
    Rx_P_opt, Ry_P_opt = clip_vector(Rx_P_opt, Ry_P_opt)
    
    # ---- Compute final fidelity & final unitary ----
    F_final, _, _, _, _, U_final = grape_grad_2channel_envelope(
        Rx_H_opt, Ry_H_opt, Rx_P_opt, Ry_P_opt,
        U_target, J, T, N,
        H_x_H, H_y_H, H_x_P, H_y_P, H_J, env
    )

    return Rx_H_opt, Ry_H_opt, Rx_P_opt, Ry_P_opt, F_final, U_final

def hard_pulse(axis, angle_deg, dt_us):
    duration = angle_deg / 180 * 80.0  # after calibration

    N = round(duration / dt_us)

    if axis.upper() == "X":
        return np.ones(N), np.zeros(N)

    if axis.upper() == "Y":
        return np.zeros(N), np.ones(N)
    
def export_to_json_GRAPE_2channel(
    filename,
    title,
    fidelity,
    totalpulsewidth,
    slices,
    Rx_H,
    Ry_H,
    Rx_P,
    Ry_P,
    owner="Unknown",
    folderpath = ""
):
    """
    Export GRAPE two-channel pulse to SpinQLab JSON format.
    Saves the file in the '/pulses/' folder.
    """

    # Ensure the pulses directory exists
    folder = "pulses/"+folderpath
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename + ".json")

    Rx_H = np.asarray(Rx_H).flatten()
    Ry_H = np.asarray(Ry_H).flatten()
    Rx_P = np.asarray(Rx_P).flatten()
    Ry_P = np.asarray(Ry_P).flatten()

    dt = totalpulsewidth / slices * 1e6

    channel1_pulses = []
    channel2_pulses = []

    for k in range(slices):

        amplitude_H = 100 * np.sqrt(Rx_H[k]**2 + Ry_H[k]**2)
        phase_H = np.degrees(np.arctan2(Ry_H[k], Rx_H[k])) % 360

        amplitude_P = 100 * np.sqrt(Rx_P[k]**2 + Ry_P[k]**2)
        phase_P = np.degrees(np.arctan2(Ry_P[k], Rx_P[k])) % 360

        channel1_pulses.append({
            "detuning": 0,
            "phase": float(phase_H),
            "amplitude": float(amplitude_H),
            "width": float(dt)
        })

        channel2_pulses.append({
            "detuning": 0,
            "phase": float(phase_P),
            "amplitude": float(amplitude_P),
            "width": float(dt)
        })

    data = {
        "description": {
            "TITLE": title,
            "OWNER": owner,
            "DATE": datetime.now().strftime("%d-%m-%Y"),
            "FIDELITY": float(fidelity),
            "TOTALPULSEWIDTH": float(totalpulsewidth),
            "SLICES": int(slices)
        },
        "parameters": {
            "offset": [{
                "channel1_pulsefre_offset": 0,
                "channel1_framefre_offset": 0,
                "channel2_pulsefre_offset": 0,
                "channel2_framefre_offset": 0
            }]
        },
        "pulse": {
            "channel1_pulse": channel1_pulses,
            "channel2_pulse": channel2_pulses
        }
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Pulse file saved to {filepath}")
    
def plot_pulse_from_json(json_file):

    with open(json_file, "r") as f:
        data = json.load(f)

    ch1 = data["pulse"]["channel1_pulse"]
    ch2 = data["pulse"]["channel2_pulse"]

    amp1 = np.array([p["amplitude"] for p in ch1])
    phase1 = np.array([p["phase"] for p in ch1])
    width1 = np.array([p["width"] for p in ch1])

    amp2 = np.array([p["amplitude"] for p in ch2])
    phase2 = np.array([p["phase"] for p in ch2])
    width2 = np.array([p["width"] for p in ch2])

    # Start time of each slice
    t = np.concatenate(([0], np.cumsum(width1[:-1])))

    # Phase colormap
    norm = mcolors.Normalize(vmin=0, vmax=360)
    cmap = plt.cm.hsv

    colors1 = cmap(norm(phase1))
    colors2 = cmap(norm(phase2))

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)

    # Channel 1
    axs[0].bar(
        t,
        amp1,
        width=width1,
        color=colors1,
        align="edge",
        edgecolor="none"
    )
    axs[0].set_ylabel("Amplitude (%)")
    axs[0].set_title("Channel 1 - Hydrogen")
    axs[0].set_xlim([0,None])
    axs[0].set_ylim([0,100])
    
    # Channel 2
    axs[1].bar(
        t,
        amp2,
        width=width2,
        color=colors2,
        align="edge",
        edgecolor="none"
    )
    axs[1].set_ylabel("Amplitude (%)")
    axs[1].set_xlabel("Time")
    axs[1].set_title("Channel 2 - Phosphorus")
    axs[1].set_xlim([0,None])
    axs[1].set_ylim([0,100])
    
    # Phase colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs)
    cbar.set_label("Phase (deg)")

    plt.show()

def load_controls_from_json(filename):

    with open(filename, "r") as f:
        data = json.load(f)

    ch1 = data["pulse"]["channel1_pulse"]
    ch2 = data["pulse"]["channel2_pulse"]
    N = data["description"]["SLICES"]
    T = data["description"]["TOTALPULSEWIDTH"]
    
    Rx_H = []
    Ry_H = []
    Rx_P = []
    Ry_P = []

    for p in ch1:
        amp = p["amplitude"] / 100.0
        phase = np.radians(p["phase"])

        Rx_H.append(amp * np.cos(phase))
        Ry_H.append(amp * np.sin(phase))

    for p in ch2:
        amp = p["amplitude"] / 100.0
        phase = np.radians(p["phase"])

        Rx_P.append(amp * np.cos(phase))
        Ry_P.append(amp * np.sin(phase))

    return (
        N,
        T,
        np.array(Rx_H),
        np.array(Ry_H),
        np.array(Rx_P),
        np.array(Ry_P)
    )

def grab_state_matrix(matrix):
    real = np.array(matrix["real"])
    imag = np.array(matrix["imag"])
    # Construct complex matrix
    rho = real + 1j * imag
    # Reshape into 4x4 matrix
    rho = rho.reshape((4,4))  
    return rho

def plot_density_matrix(rho):

    n = rho.shape[0]

    real = np.real(rho)
    imag = np.imag(rho)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    xpos, ypos = np.meshgrid(np.arange(n), np.arange(n))
    xpos = xpos.flatten()
    ypos = ypos.flatten()

    zpos = np.zeros(n*n)

    dx = dy = 0.35

    real_vals = real.flatten()
    imag_vals = imag.flatten()

    # shift bars slightly left/right
    xpos_real = xpos - 0.2
    xpos_imag = xpos + 0.2

    # colors
    real_colors = ["blue" if v >= 0 else "cyan" for v in real_vals]
    imag_colors = ["red" if v >= 0 else "orange" for v in imag_vals]

    # real bars
    ax.bar3d(
        xpos_real,
        ypos,
        zpos,
        dx,
        dy,
        real_vals,
        color=real_colors,
        shade=True,
        label="Real"
    )

    # imaginary bars
    ax.bar3d(
        xpos_imag,
        ypos,
        zpos,
        dx,
        dy,
        imag_vals,
        color=imag_colors,
        shade=True,
        label="Imag"
    )

    labels = ['00','01','10','11'][:n]
    
    # Clean grid
    ax.grid(True)

    ax.xaxis._axinfo["grid"]['color'] = (0.8,0.8,0.8,0.4)
    ax.yaxis._axinfo["grid"]['color'] = (0.8,0.8,0.8,0.4)
    ax.zaxis._axinfo["grid"]['color'] = (0.8,0.8,0.8,0.4)

    # Remove wall shading
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    ax.set_xlabel("Column", labelpad=15)
    ax.set_ylabel("Row", labelpad=15)
    ax.set_zlabel("Value", labelpad=10)

    ax.set_xticklabels(labels, rotation=30)
    ax.set_yticklabels(labels, rotation=-30)

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.9)

    ax.set_title("Density Matrix")
    
    ax.view_init(elev=25, azim=-55)

    # ax.set_zlim(-1,1)

    # plt.tight_layout()
    plt.show()
    
def plot_density_matrix_separate(rho):

    n = rho.shape[0]

    real = np.real(rho)
    imag = np.imag(rho)

    fig = plt.figure(figsize=(12,5))

    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    axes = [ax1, ax2]

    labels = ['00','01','10','11'][:n]

    for ax, matrix, title in zip(
        axes,
        [real, imag],
        ["Real part", "Imaginary part"]
    ):

        xpos, ypos = np.meshgrid(np.arange(n), np.arange(n))
        xpos = xpos.flatten()
        ypos = ypos.flatten()

        values = matrix.flatten()

        colors = ["red" if v < 0 else "blue" for v in values]

        ax.bar3d(
            xpos,
            ypos,
            np.zeros_like(values),
            0.5,
            0.5,
            values,
            color=colors,
            shade=True
        )

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))

        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_zlabel("Value")
        
        ax.set_zlim(0,1)
        ax.set_title(title)
        ax.view_init(elev=25, azim=-55)

    plt.show()
    
def parse_pulse_experiments(filepath: str, pulse_name: str) -> list[dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        log_text = f.read()

    # Split on any pulse header, keeping the delimiter
    pulse_header = r'(?:NEW\s+)?(?:PULSE|PULSO):?\s*(?!WIDTH)'
    blocks = re.split(rf'({pulse_header}\S+.*?\n)', log_text)
    
    # Find the block whose header matches pulse_name exactly
    pulse_block = None
    for i, chunk in enumerate(blocks):
        name_match = re.match(rf'{pulse_header}(\S+)', chunk.strip())
        if name_match and name_match.group(1).strip() == pulse_name:
            # The content follows immediately after the header
            pulse_block = blocks[i + 1] if i + 1 < len(blocks) else ""
            break

    if pulse_block is None:
        print(f"Pulse '{pulse_name}' not found in '{filepath}'.")
        return []

    results = []
    for block in re.split(r'Estado inicial:', pulse_block)[1:]:
        state_match    = re.search(r'(\|[01]{2}>)', block)
        expected_match = re.search(r'Estado esperado:\s*(\|[01]{2}>)', block)
        matrix_match   = re.search(r"matrix:\s*(\{.*?\})\s*\n", block)
        prob_match     = re.search(r"probability:\s*(\[.*?\])\s*\n", block)

        if not all([state_match, expected_match, matrix_match, prob_match]):
            continue

        results.append({
            "initial_state":  state_match.group(1),
            "expected_state": expected_match.group(1),
            "matrix":         ast.literal_eval(matrix_match.group(1)),
            "probability":    ast.literal_eval(prob_match.group(1)),
        })

    return results

def parse_swap_test_experiments(filepath: str, pulse_name: str) -> list[dict]:
    """
    Parse SWAP test log entries. Handles state labels like |+0>, |00>+|10>, etc.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        log_text = f.read()

    pulse_header = r'PULSE:\s*'
    pulse_pattern = rf'{pulse_header}{re.escape(pulse_name)}(?!\S)\s*\n(.*?)(?={pulse_header}|\Z)'
    pulse_match = re.search(pulse_pattern, log_text, re.DOTALL)

    if not pulse_match:
        print(f"Pulse '{pulse_name}' not found in '{filepath}'.")
        return []

    block = pulse_match.group(1)

    state_match    = re.search(r'Estado inicial:\s*(\S+)', block)
    expected_match = re.search(r'Estado esperado:\s*(\S+)', block)
    matrix_match   = re.search(r"matrix:\s*(\{.*?\})\s*\n", block)
    prob_match     = re.search(r"probability:\s*(\[.*?\])\s*\n", block)

    if not all([state_match, expected_match, matrix_match, prob_match]):
        print(f"Could not fully parse block for '{pulse_name}'.")
        return []

    return [{
        "initial_state":  state_match.group(1),
        "expected_state": expected_match.group(1),
        "matrix":         ast.literal_eval(matrix_match.group(1)),
        "probability":    ast.literal_eval(prob_match.group(1)),
    }]

def get_state(label: str) -> np.ndarray:
    """Return the state vector for a single-qubit label: '0', '1', '+', or '-'."""
    if label not in SINGLE_STATES:
        raise ValueError(f"Unknown state label '{label}'. Must be one of {list(SINGLE_STATES.keys())}.")
    return SINGLE_STATES[label]

def calc_CNOT_fidelity(experiments: list[dict]) -> float:
    """
    Calculate experimental gate fidelity from the 4 state experiments.
    
    Picks the probability of the expected output state for each experiment:
        |00> -> |00>  : prob[0]
        |10> -> |11>  : prob[3]
        |01> -> |01>  : prob[1]
        |11> -> |10>  : prob[2]

    Args:
        experiments: Output of parse_pulse_experiments (must contain all 4 states).

    Returns:
        Gate fidelity as a float.
    """
    # Map each expected state to its index in the probability array
    state_to_index = {"|00>": 0, "|01>": 1, "|10>": 2, "|11>": 3}

    if len(experiments) != 4:
        raise ValueError(f"Expected 4 experiments, got {len(experiments)}.")

    total = 0.0
    for exp in experiments:
        expected = exp["expected_state"]
        idx = state_to_index[expected]
        total += exp["probability"][idx]

    fidelity = total / 4
    print(f"Experimental Gate Fidelity: {fidelity:.6f}")
    return fidelity

def calc_fidelity_from_matrix(experiments: list[dict], U_target: np.ndarray) -> float:
    """
    Calculate experimental gate fidelity using density matrix trace overlap.
    
    For each initial state |i>, constructs the experimental density matrix rho
    from the measured matrix, then computes:
    
        F = (1/N) * sum_i Tr(rho_i_exp @ rho_i_ideal)
    
    where rho_i_ideal = U_target |i><i| U_target†
    
    Args:
        experiments: Output of parse_pulse_experiments.
        U_target:    Ideal target unitary (d x d numpy array).
    
    Returns:
        Gate fidelity as a float.
    """
    # Map state label to its computational basis index
    state_to_index = {
        "|00>": 0, "|01>": 1, "|10>": 2, "|11>": 3,
        "|000>": 0, "|001>": 1, "|010>": 2, "|011>": 3,
        "|100>": 4, "|101>": 5, "|110>": 6, "|111>": 7,
    }

    d = U_target.shape[0]
    total = 0.0

    for exp in experiments:
        # Reconstruct experimental density matrix from log
        real = np.array(exp["matrix"]["real"]).reshape(d, d)
        imag = np.array(exp["matrix"]["imag"]).reshape(d, d)
        rho_exp = real + 1j * imag

        # Build ideal output density matrix: rho_ideal = U |i><i| U†
        i = state_to_index[exp["initial_state"]]
        psi_in = np.zeros(d, dtype=np.complex128)
        psi_in[i] = 1.0
        psi_out = U_target @ psi_in                      # ideal output state
        rho_ideal = np.outer(psi_out, psi_out.conj())    # |psi_out><psi_out|

        # Trace overlap
        overlap = np.real(np.trace(rho_ideal @ rho_exp))
        total += overlap

    fidelity = total / len(experiments)
    print(f"Experimental Gate Fidelity (trace): {fidelity:.6f}")
    return fidelity

def calc_swap_test_overlap(probability: list[float]) -> float:
    """
    Compute |<psi|phi>|^2 from a destructive SWAP test measurement.
    
    The SWAP test circuit (CNOT + Hadamard on control) maps:
        P(|11>) = (1 - |<psi|phi>|^2) / 2
    
    So:
        |<psi|phi>|^2 = 1 - 2*P(|11>)
    
    Args:
        probability: list of 4 floats [P(00), P(01), P(10), P(11)]
    
    Returns:
        overlap: |<psi|phi>|^2, the squared scalar product between the two input states.
    """
    p_11 = probability[3]
    overlap = 1 - 2 * p_11
    
    print(f"P(|11>) = {p_11:.6f}")
    print(f"|<psi|phi>|^2 = {overlap:.6f}")
    
    return overlap

def expected_overlap(initial_state: str) -> float:
    """
    Compute the theoretical |<psi|phi>|^2 for a SWAP test initial state string.
    
    Args:
        initial_state: e.g. '|+0>', '|01>', '|+->', etc. 
                        First character = psi, second character = phi.
    
    Returns:
        |<psi|phi>|^2 as a float.
    """
    # Strip the |...> notation
    labels = initial_state.strip("|>")
    
    if len(labels) != 2:
        raise ValueError(f"Expected exactly 2 single-qubit labels in '{initial_state}', got '{labels}'.")
    
    psi_label, phi_label = labels[0], labels[1]
    psi = get_state(psi_label)
    phi = get_state(phi_label)
    
    inner_product = np.vdot(psi, phi)  # <psi|phi>
    overlap = np.abs(inner_product)**2
    
    print(f"|<{psi_label}|{phi_label}>|^2 = {overlap:.6f}")
    
    return overlap


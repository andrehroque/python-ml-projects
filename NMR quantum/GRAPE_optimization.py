import numpy as np
import scipy
import json
import matplotlib.pyplot as plt
from datetime import datetime
from qiskit.quantum_info import Pauli

np.set_printoptions(linewidth=200, precision=4, suppress=True)

# Initialize Pauli matrices

pauli_x = Pauli('X')
pauli_y = Pauli('Y')
pauli_z = Pauli('Z')
identity = Pauli('I')

X = pauli_x.to_matrix()
Y = pauli_y.to_matrix()
Z = pauli_z.to_matrix()
I = identity.to_matrix()

Ix1 = np.kron(X/2,I)
Iy1 = np.kron(Y/2,I)
Iz1 = np.kron(Z/2,I)
Ix2 = np.kron(I,X/2)
Iy2 = np.kron(I,Y/2)
Iz2 = np.kron(I,Z/2)

U_cnot = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,0,1],
                   [0,0,1,0]])

U_perm = np.array([[1,0,0,0],
                   [0,0,0,1],
                   [0,1,0,0],
                   [0,0,1,0]])

U_cz = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,0],
                 [0,0,0,-1]])

# Gradient Ascent - GRAPE - different channels

# a bit different from the basic SMP, the parameter phi is omitted, collapsed into the amplitude which now differs for each orthogonal direction x, y

# constants
n = 2
d = 2**n
#wH = 27e6
#wP = 11e6
#w1H = w1P after calibration
w1_max = 6250

H_x_H = 2*np.pi*w1_max*(Ix1)
H_x_P = 2*np.pi*w1_max*(Ix2)
H_y_H = 2*np.pi*w1_max*(Iy1)
H_y_P = 2*np.pi*w1_max*(Iy2)

# H_0 = 0 because on resonance, only include when detuning or counteracting chemical shift mismatch

def p_calc_2channel(Rx_H, Ry_H, Rx_P, Ry_P, U_target, J, T, N):
    P_list = []
    P = U_target.copy()
    dt = T/N
    H_J = 2*np.pi*J*(np.kron(Z,Z)/4)
    
    for k in reversed(range(N)):
        H = H_J + Rx_H[k]*H_x_H + Ry_H[k]*H_y_H + Rx_P[k]*H_x_P + Ry_P[k]*H_y_P
        Uk = scipy.linalg.expm(-1j * H * dt)
        P = Uk.conj().T @ P
        P_list.insert(0, P)

    return P_list

def x_calc_2channel(Rx_H, Ry_H, Rx_P, Ry_P, J, T, N):
    X_list = []
    U = np.eye(d, dtype=complex)
    dt = T/N
    H_J = 2*np.pi*J*(np.kron(Z,Z)/4)
    
    for k in range(N):
        H = H_J + Rx_H[k]*H_x_H + Ry_H[k]*H_y_H + Rx_P[k]*H_x_P + Ry_P[k]*H_y_P
        Uk = scipy.linalg.expm(-1j * H * dt)
        U = Uk @ U
        X_list.append(U)

    return X_list, U

def fidelity_grape(U, U_target):
    phi = np.trace(U_target.conj().T @ U)
    return np.abs(phi)**2/d**2

def grape_grad_2channel(Rx_H, Ry_H, Rx_P, Ry_P, U_target, J, T, N):
    
    X_list, U_final = x_calc_2channel(Rx_H, Ry_H, Rx_P, Ry_P, J, T, N)
    P_list = p_calc_2channel(Rx_H, Ry_H, Rx_P, Ry_P, U_target, J, T, N)
    dt = T/N
    
    phi = np.trace(U_target.conj().T @ U_final)
    F0 = fidelity_grape(U_final, U_target)
    
    grad_Rx_H = np.zeros(N)
    grad_Ry_H = np.zeros(N)
    grad_Rx_P = np.zeros(N)
    grad_Ry_P = np.zeros(N)
    
    for k in range(N):
        Xk = X_list[k]
        Pk = P_list[k]
        
        dX_Rx_H = -1j * dt * H_x_H @ Xk
        dX_Ry_H = -1j * dt * H_y_H @ Xk
        dX_Rx_P = -1j * dt * H_x_P @ Xk
        dX_Ry_P = -1j * dt * H_y_P @ Xk

        grad_Rx_H[k] = 2 * np.real(
            np.trace(Pk.conj().T @ dX_Rx_H) * np.conj(phi)
        )

        grad_Ry_H[k] = 2 * np.real(
            np.trace(Pk.conj().T @ dX_Ry_H) * np.conj(phi)
        )

        grad_Rx_P[k] = 2 * np.real(
            np.trace(Pk.conj().T @ dX_Rx_P) * np.conj(phi)
        )

        grad_Ry_P[k] = 2 * np.real(
            np.trace(Pk.conj().T @ dX_Ry_P) * np.conj(phi)
        )
        
    return (F0, grad_Rx_H, grad_Ry_H, grad_Rx_P, grad_Ry_P, U_final)

def pulse_optimize_grape(U_target, J, T, N, learn_rate, max_iter, startParameters = None):
    
    print("Target matrix: \n", U_target)

    dt = T/N

    #print("dt = ",dt) # debug

    # initialize
    
    if startParameters is not None:
        Rx_H,Ry_H,Rx_P,Ry_P = startParameters
    else:
        Rx_H = 0.2 * np.random.rand(N)
        Ry_H = 0.2 * np.random.rand(N)
        Rx_P = 0.2 * np.random.rand(N)
        Ry_P = 0.2 * np.random.rand(N)

    lr = learn_rate
    maxi = max_iter

    for i in range(maxi):

        F, gRxH, gRyH, gRxP, gRyP, Ufinal = grape_grad_2channel(
            Rx_H, Ry_H, Rx_P, Ry_P, U_target
        )

        Rx_H += lr * gRxH
        Ry_H += lr * gRyH
        Rx_P += lr * gRxP
        Ry_P += lr * gRyP

        # Clip amplitudes
        Rx_H = np.clip(Rx_H, -1, 1)
        Ry_H = np.clip(Ry_H, -1, 1)
        Rx_P = np.clip(Rx_P, -1, 1)
        Ry_P = np.clip(Ry_P, -1, 1)

        if i % 100 == 0:
            print(f"Iter {i}, Fidelity = {F:.6f}")
        
        if F > 0.999:
            print("Converged.")
            break
    
    print("Final fidelity:", F)
    print("Final matrix: \n", Ufinal)
    
def export_to_json_GRAPE_2channel(filename, title, fidelity, totalpulsewidth, slices, Rx_H, Ry_H, Rx_P, Ry_P, owner="Unkown"):
    
    Rx_H = np.asarray(Rx_H).flatten()
    Ry_H = np.asarray(Ry_H).flatten()
    Rx_P = np.asarray(Rx_P).flatten()
    Ry_P = np.asarray(Ry_P).flatten()
    
    dt = totalpulsewidth / slices

    channel1_pulses = []
    channel2_pulses = []

    for k in range(slices):

        amplitude_H = 100 * np.sqrt(Rx_H[k]**2 + Ry_H[k]**2)  # percent
        phase_H = np.degrees(np.arctan2(Ry_H[k], Rx_H[k])) % 360

        amplitude_P = 100 * np.sqrt(Rx_P[k]**2 + Ry_P[k]**2)
        phase_P = np.degrees(np.arctan2(Ry_P[k], Rx_P[k])) % 360
        
        pulse_H = {
            "detuning": 0,
            "phase": float(phase_H),
            "amplitude": float(amplitude_H),
            "width": float(dt)
        }
        pulse_P = {
            "detuning": 0,
            "phase": float(phase_P),
            "amplitude": float(amplitude_P),
            "width": float(dt)
        }

        # Same pulse on both channels (modify if needed)
        channel1_pulses.append(pulse_H)
        channel2_pulses.append(pulse_P)
        
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
            "offset": [
                {
                    "channel1_pulsefre_offset": 0,
                    "channel1_framefre_offset": 0,
                    "channel2_pulsefre_offset": 0,
                    "channel2_framefre_offset": 0
                }
            ]
        },
        "pulse": {
            "channel1_pulse": channel1_pulses,
            "channel2_pulse": channel2_pulses
        }
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Pulse file saved to {filename}")
    
    
    
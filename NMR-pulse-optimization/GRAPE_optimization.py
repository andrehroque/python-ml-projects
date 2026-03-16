import numpy as np
import scipy
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
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

# H_0 = 0 because working on resonance. Only include when detuning or counteracting chemical shift mismatch

def p_calc_2channel(Rx_H, Ry_H, Rx_P, Ry_P, U_target, J, T, N):
    """
    GRAPE algorithm backwards propagation calculation.
    """        
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
    """
    GRAPE algorithm forwards propagation calculation.
    """    
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
    """
    GRAPE algorithm gradient calculation.
    """
    
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
    
    """
    Run optimization loop up to max iterations or theoretical convergence: fidelity >= 0.999.
    """
    
    #print("Target matrix: \n", U_target)

    dt = T/N

    #print("dt = ",dt) # debug

    # initialize
    
    if startParameters is not None:
        Rx_H,Ry_H,Rx_P,Ry_P = startParameters
    else:
        Rx_H = np.full(N, 0.2)
        Ry_H = np.full(N, 0.2)
        Rx_P = np.full(N, 0.2)
        Ry_P = np.full(N, 0.2)

    lr = learn_rate
    maxi = max_iter

    for i in range(maxi):

        F, gRxH, gRyH, gRxP, gRyP, Ufinal = grape_grad_2channel(
            Rx_H, Ry_H, Rx_P, Ry_P, U_target, J, T, N
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

        #if i % 100 == 0:
        #    print(f"Iter {i}, Fidelity = {F:.6f}")
        
        if F > 0.999:
            print("Converged.")
            break
    
    print("Final fidelity:", F)
    #print("Final matrix: \n", Ufinal)
    return Rx_H, Ry_H, Rx_P, Ry_P, F

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
    owner="Unknown"
):
    """
    Export GRAPE two-channel pulse to SpinQLab JSON format.
    """

    Rx_H = np.asarray(Rx_H).flatten()
    Ry_H = np.asarray(Ry_H).flatten()
    Rx_P = np.asarray(Rx_P).flatten()
    Ry_P = np.asarray(Ry_P).flatten()

    dt = totalpulsewidth / slices

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

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Pulse file saved to {filename}")
    
def plot_pulse_from_json(json_file):
    """
    Import pulse data from SpinQLab JSON format to plot.
    """

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
    axs[0].set_xlim([None,None])
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
    axs[1].set_xlim([None,None])
    axs[1].set_ylim([0,100])
    
    # Phase colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs)
    cbar.set_label("Phase (deg)")

    plt.show()

def grab_state_matrix(matrix):
    """
    Edit SpinQLab Link output matrix from dictionary to numpy array matrix.
    """
    
    real = np.array(matrix["real"])
    imag = np.array(matrix["imag"])
    # Construct complex matrix
    rho = real + 1j * imag
    # Reshape into 4x4 matrix
    rho = rho.reshape((4,4))  
    return rho
    
def plot_density_matrix(rho):
    """
    Plot numpy array matrix for resulting state density matrices.
    """

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
    
    
    
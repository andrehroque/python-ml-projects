import json, os, numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors
from spinqlablink import Pulse
from datetime import datetime

def load_pulses_from_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    pulses = []

    ch1 = data["pulse"]["channel1_pulse"]
    ch2 = data["pulse"]["channel2_pulse"]

    for p1, p2 in zip(ch1, ch2):
        pulses.append(Pulse(path=0, phase=p1["phase"], amplitude=p1["amplitude"], width=p1["width"], detuning=p1.get("detuning", 0)))
        pulses.append(Pulse(path=1, phase=p2["phase"], amplitude=p2["amplitude"], width=p2["width"], detuning=p2.get("detuning", 0)))

    return pulses

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

    # Phase colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs)
    cbar.set_label("Phase (deg)")

    plt.show()

def write_log(log_file: str, pulse_json_path: str, initial_state: str, expected_state: str, exp_result: dict):
    """
    Append experiment result to a log file in the standard format.

    Args:
        log_file:         Path to the output .txt log file.
        pulse_json_path:  Path to the pulse JSON file.
        initial_state:    e.g. '|00>', '|10>', etc.
        expected_state:   e.g. '|00>', '|11>', etc.
        exp_result:       The result dict from spinqlablink.get_experiment_result()['result']
    """
    pulse_name = os.path.splitext(os.path.basename(pulse_json_path))[0]

    with open(pulse_json_path, "r") as f:
        pulse_data = json.load(f)
    desc = pulse_data["description"]
    fidelity        = desc["FIDELITY"]
    totalpulsewidth = desc["TOTALPULSEWIDTH"]
    slices          = desc["SLICES"]

    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        if initial_state == "|00>":
            f.write(f"\n{'-' * 102}\n\n")
            f.write(f"PULSE: {pulse_name}\n\n")
            f.write(f"FIDELITY: {fidelity}\n")
            f.write(f"TOTALPULSEWIDTH: {totalpulsewidth}\n")
            f.write(f"SLICES: {slices}\n")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        f.write(f"\nEstado inicial: {initial_state}\n")
        f.write(f"Estado esperado: {expected_state}\n\n")

        for key, value in exp_result.items():
            if key != "graph":
                f.write(f"{key}: {value}\n")

        f.write("Chart data plotting completed\n")
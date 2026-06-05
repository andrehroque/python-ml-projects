from spinqlablink import SpinQLabLink, ExperimentType, Pulse, Gradient, Gate, CustomGate, Circuit
from spinqlablink import print_graph
from funcs import load_pulses_from_json, write_log
import json, os, numpy as np, matplotlib.pyplot as plt, matplotlib.colors as mcolors
from datetime import datetime

def main():
    # Create connection
    
    # DF ID: 192.168.4.34
    # DEEC ID: 10.206.88.204
    spinqlablink = SpinQLabLink("10.206.88.204", 8181, "andreroque", "anyword")
    spinqlablink.connect()

    if not spinqlablink.wait_for_login():
        print("Login failed")
        return

    exp_circuit_layer, exp_circuit_layer_para = spinqlablink.register_experiment(ExperimentType.CIRCUIT_LAYER_EXPERIMENT)
    
    PULSE_JSON     = "pulses/CNOT/CNOT_tstudy3.json"
    LOG_FILE       = "data/CNOT_logs.txt"
    INITIAL_STATE  = "|11>" # Change this to test gate for each initial state
    CNOT_TRUTH_TABLE = {
    "|00>": "|00>",
    "|01>": "|01>",
    "|10>": "|11>",
    "|11>": "|10>",
    }
    EXPECTED_STATE = CNOT_TRUTH_TABLE[INITIAL_STATE]
    STATE_PREP = {
    "|00>": [],
    "|01>": [Pulse(path=1, width=80, amplitude=100, phase=90, detuning=0)],
    "|10>": [Pulse(path=0, width=80, amplitude=100, phase=90, detuning=0)],
    "|11>": [Pulse(path=1, width=80, amplitude=100, phase=90, detuning=0),
             Pulse(path=0, width=80, amplitude=100, phase=90, detuning=0)],
    }
    # Circuit layer experiment creates pps automatically

    using_pulse = True
    # pulse_sequence = []
    # load_pulses_from_json parses json file to array of pulse objects
    
    if using_pulse:
        pulse_sequence = load_pulses_from_json(PULSE_JSON)
        # pulse_sequence = []
        pulse_sequence = STATE_PREP[INITIAL_STATE] + pulse_sequence
    else:
        pulse_sequence = load_pulses_from_json("pulses/CNOT_lbfgs_optimized_2.json")
    
    # with open("pulses/pulseCNOT_0.2.json", "r") as f:
    #     data = json.load(f)
    # gate_from_json = json.dumps(data)

    if using_pulse:
        exp_circuit_layer_para.pulses = pulse_sequence
        # exp_circuit_layer_para.pulses = [Pulse(path=0, width = 80, amplitude = 100, phase = 90, detuning = 0)]
    else:
        circuit = Circuit(2)
        # circuit << Gate(type='H', qubitIndex=0)
        circuit << Gate(type='X', qubitIndex=1)
        circuit << Gate(type='X', qubitIndex=0)
        circuit << CustomGate(type='CNOT', customType='andre_custom_gate', pulses=pulse_sequence)
        # circuit << CustomGate(type='I', customType='andre_custom_gate', controlQubit=0, qubitIndex=1, gateJson=gate_from_json)
        # circuit << CustomGate(type='I', customType='I_custom_gate', qubitIndex=1, pulses=[Pulse(path=0, width = 80, amplitude = 100, phase = 90, detuning = 0)])
        circuit.print_circuit()
        exp_circuit_layer_para.set_circuit(circuit)

    # Set other parameters
    exp_circuit_layer_para.using_pulse = using_pulse # True: using pulse to testing gate, False: using circuit to get result
    exp_circuit_layer_para.gradients = [Gradient(delay=0,value=0)] # gradient sequence
    exp_circuit_layer_para.relaxation_delay = 15 #S
    exp_circuit_layer_para.h_freShift = 0 # hydrogen frequency shift(Hz)
    exp_circuit_layer_para.p_freShift = 0 # phosphorus frequency shift(Hz)
    exp_circuit_layer_para.h_freDemo = 0 # hydrogen frequency demo(Hz)
    exp_circuit_layer_para.p_freDemo = 0 # phosphorus frequency demo(Hz)
    exp_circuit_layer_para.sampleFre = 10000 # sample frequency(Hz)
    exp_circuit_layer_para.sampleCount = 16000 # sample count
    exp_circuit_layer_para.sampleDelay = 0 # sample delay(us)
    exp_circuit_layer_para.samplePath = -1 # Sampling path: 0=Hydrogen channel, 1=Phosphorus channel, -1=both channels

    spinqlablink.run_experiment()
    print("Waiting for experiment completion")
    spinqlablink.wait_for_experiment_completion()

    exp_info = spinqlablink.get_experiment_result()
    spinqlablink.deregister_experiment()
    spinqlablink.disconnect()

    if "result" in exp_info:
        exp_result = exp_info["result"]
        for key, value in exp_result.items():
            if key != "graph":
                print(f"{key}: {value}")
        print_graph(exp_info["result"])
        write_log(LOG_FILE, PULSE_JSON, INITIAL_STATE, EXPECTED_STATE, exp_result)
    else:
        print("Experiment failed")

if __name__ == "__main__":
    main()
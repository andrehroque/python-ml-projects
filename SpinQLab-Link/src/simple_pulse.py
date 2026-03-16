from spinqlablink import SpinQLabLink, ExperimentType, Pulse
from spinqlablink import print_graph
from funcs import load_pulses_from_json

def main():
    # Establish connection
    spinqlablink = SpinQLabLink("192.168.4.34", 8181, "andreroque", "password")
    spinqlablink.connect()
    spinqlablink.wait_for_login()
    
    try:
        # # Connect and login
        # if not spinqlablink.connect():
        #     raise Exception("Connection failed")

        # if not spinqlablink.wait_for_login():
        #     raise Exception("Login failed")

        print("✓ Connection successful")

        # Register experiment
        experiment, parameters = spinqlablink.register_experiment(
            ExperimentType.NMR_PHENOMENON_AND_SIGNAL
        )
        print(f"✓ Experiment registered: {experiment.id}")
        
        pulse_sequence = load_pulses_from_json("pulses\pulseCNOT_0.4.json")
        pulse_sequence.insert(0,Pulse(path=0, width=40, amplitude=100, phase=90, detuning=0))
        # Set parameters
        parameters.pulses = pulse_sequence
        parameters.freq_h = 27.852105
        parameters.freq_p = 11.322872
        parameters.makePps = True
        parameters.samplePath = 1
        parameters.custom_freq = False
        print("✓ Parameters set")

        # Run experiment
        spinqlablink.run_experiment()
        print("✓ Experiment started")

        # Wait for completion
        spinqlablink.wait_for_experiment_completion()
        print("✓ Experiment completed")

        # Get results
        result = spinqlablink.get_experiment_result()
        print(f"✓ Experiment status: {result.get('status', 'finished')}")

        if result:
            print_graph(result["result"])
        else:
            print("✗ Experiment timed out")

    except Exception as e:
        print(f"✗ Error: {e}")

    finally:
        # Clean up resources
        spinqlablink.deregister_experiment()
        spinqlablink.disconnect()
        print("✓ Resources cleaned up")

if __name__ == "__main__":
    main()
from spinqlablink import SpinQLabLink, ExperimentType, Pulse

def main():
    # Establish connection
    spinqlablink = SpinQLabLink("192.168.9.121", 8181, "andreroque", "password")

    try:
        # Connect and login
        if not spinqlablink.connect():
            raise Exception("Connection failed")

        if not spinqlablink.wait_for_login():
            raise Exception("Login failed")

        print("✓ Connection successful")

        # Register experiment
        experiment, parameters = spinqlablink.register_experiment(
            ExperimentType.NMR_PHENOMENON_AND_SIGNAL
        )
        print(f"✓ Experiment registered: {experiment.id}")

        # Set parameters
        parameters.pulses = [Pulse(path=0, width=40, amplitude=100, phase=90, detuning=0)]
        parameters.freq_h = 37.852105
        parameters.freq_p = 15.322872
        parameters.makePps = True
        parameters.samplePath = 0
        parameters.custom_freq = False
        print("✓ Parameters set")

        # Run experiment
        spinqlablink.run_experiment()
        print("✓ Experiment started")

        # Wait for completion
        if spinqlablink.wait_for_experiment_completion():
            print("✓ Experiment completed")

            # Get results
            result = spinqlablink.get_experiment_result()
            print(f"✓ Experiment status: {result['status']}")

            # Display results (optional)
            # from examples.toolsfunc import print_graph
            # print_graph(result["result"])

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
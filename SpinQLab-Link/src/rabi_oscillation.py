import time
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from spinqlablink import SpinQLabLink, ExperimentType, Pulse


def rabi_oscillation_experiment():
    """Run a Rabi oscillation experiment using SpinQLabLink."""
    
    spinqlablink = SpinQLabLink("192.168.9.134", 0000, "andreroque", "password")
    spinqlablink.connect()
    spinqlablink.wait_for_login()
    
    # Pulse width range
    width_list = [i * 40 for i in range(10)]  # 0, 40, 80, ..., 360 s
    intensities = []
    
    try:
        for width in width_list:
            print(f"Testing pulse width: {width} s")
            # Register experiment
            _, exp_para = spinqlablink.register_experiment(ExperimentType.RABI_OSCILLATIONS)
            # Set parameters
            exp_para.freq_h = 37.852105
            exp_para.freq_p = 15.322872
            exp_para.makePps = True
            exp_para.samplePath = 0
            exp_para.custom_freq = False
            exp_para.pulses = [Pulse(path=0, width=width, amplitude=100, phase=90, detuning=0)]
            # Run experiment
            spinqlablink.run_experiment()
            spinqlablink.wait_for_experiment_completion()
            
            # Get results
            result = spinqlablink.get_experiment_result()
            signal_data = result["result"]["real"]
            
            # Calculate signal intensity (FFT peak)
            if signal_data:
                intensities.append(signal_data)
            else:
                intensities.append(0)
            
            spinqlablink.deregister_experiment()
            time.sleep(2)  # Wait for system stabilization

    except Exception as e:
        print(f"Experiment failed: {e}")

    # Fit Rabi oscillation curve
    def rabi_function(x, A, B, freq, phase, decay):
        return A * np.cos(2 * np.pi * freq * x + phase) * np.exp(-x / decay) + B

    try:
        p0 = [4000, 0, 1/280, np.pi, 500]
        popt, _ = curve_fit(rabi_function, width_list, intensities, p0=p0)
        pi_pulse_width = np.pi / (2 * np.pi * popt[2])
        print(f"Estimated π pulse width: {pi_pulse_width:.1f} s")

        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(width_list, intensities, 'bo-', label='Experimental data')
        x_fit = np.linspace(min(width_list), max(width_list), 1000)
        y_fit = rabi_function(x_fit, *popt)
        plt.plot(x_fit, y_fit, 'r-', label='Fitted curve')
        plt.xlabel('Pulse width (s)')
        plt.ylabel('Signal intensity')
        plt.title('Rabi Oscillation')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    except Exception as e:
        print(f"Fitting failed: {e}")
        
    finally:
        spinqlablink.disconnect()


if __name__ == "__main__":
    rabi_oscillation_experiment()
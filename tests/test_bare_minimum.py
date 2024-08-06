import os
import sys

sys.path.insert(1, os.getcwd() + "/../src")


def testing_schurmann_bit_generation_interface():
    import numpy as np

    from signal_processing.shurmann import SchurmannProcessing

    outcome = SchurmannProcessing.zero_out_antialias_sigs_algo(
        np.sin(np.arange(50000)),
        18000,
        24000,
        10000,
        1000,
    )

    assert type(outcome) is bytes  # nosec


def testing_miettinen_bit_generation_interface():
    import numpy as np

    from signal_processing.miettinen import MiettinenProcessing

    outcome = MiettinenProcessing.miettinen_algo(
        np.sin(np.arange(50000)), 1000, 1000, 0.5, 0.5
    )

    assert type(outcome) is bytes  # nosec


def testing_perceptio_bit_generation_interface():
    import numpy as np

    from signal_processing.perceptio import PerceptioProcessing

    rng = np.random.default_rng(0)
    signal = rng.integers(0, 10, size=100000)

    fps, events = PerceptioProcessing.perceptio(signal, 4, 48000, 0.75, 2, 0.1, 2, 5, 3)

    assert type(fps) is list  # nosec
    assert type(fps[0]) is bytes  # nosec


def testing_iotcupid_bit_generation_interface():
    import numpy as np

    from signal_processing.iotcupid import IoTCupidProcessing

    rng = np.random.default_rng(0)
    signal = rng.integers(0, 10, size=100000)

    fps, events = IoTCupidProcessing.iotcupid(
        signal, 128, 10000, 0.75, 4, 3, 1, 0.1, 10, 0.05, 0.07, 4, 1.1, 2, 10, 0.8
    )

    assert type(fps) is list  # nosec
    assert type(fps[0]) is bytes  # nosec


def testing_fastzip_algorithm():
    import numpy as np

    from signal_processing.fastzip import FastZIPProcessing

    # Simulate sensor data, e.g., 3D accelerometer data
    sample_rate = 100
    signal = np.random.normal(0, 1, sample_rate)  # Simulated 1-second 1D data

    # Parameters configuration for FastZIP
    n_bits = 12
    power_threshold = -12
    snr_threshold = 1.2
    number_peaks = 0
    bias = 0
    eqd_delta = 1
    peak_status = True
    ewma_filter = True
    alpha = 0.015
    remove_noise = False
    normalize = True

    # Prepare list of sensor data and parameters for testing fastzip_algo
    sensor_data_list = [signal]
    n_bits_list = [n_bits]
    power_thresh_list = [power_threshold]
    snr_thresh_list = [snr_threshold]
    peak_thresh_list = [number_peaks]
    bias_list = [bias]
    sample_rate_list = [sample_rate]
    eqd_delta_list = [eqd_delta]
    peak_status_list = [peak_status]
    ewma_filter_list = [ewma_filter]
    alpha_list = [alpha]
    remove_noise_list = [remove_noise]
    normalize_list = [normalize]

    # Execute fastzip_algo
    key = FastZIPProcessing.fastzip_algo(
        sensor_data_list,
        n_bits_list,
        power_thresh_list,
        snr_thresh_list,
        peak_thresh_list,
        bias_list,
        sample_rate_list,
        eqd_delta_list,
        peak_status_list,
        ewma_filter_list,
        alpha_list,
        remove_noise_list,
        normalize_list,
    )

    assert type(key) is bytes  # nosec

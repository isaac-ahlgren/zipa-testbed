import os
import sys

sys.path.insert(1, os.getcwd() + "/src")


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
        signal,
        128,
        10000,
        0.75,
        4,
        3,
        1,
        0.1,
        10,
        0.05,
        0.07,
        4,
        1.1,
        2,
        10
    )

    assert type(fps) is list  # nosec
    assert type(fps[0]) is bytes  # nosec


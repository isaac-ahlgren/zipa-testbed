import os
import sys

sys.path.insert(1, os.getcwd() + "/src")


def testing_schurmann_bit_generation_interface():
    import numpy as np

    from protocols.shurmann import Shurmann_Siggs_Protocol

    outcome = Shurmann_Siggs_Protocol.zero_out_antialias_sigs_algo(
        np.sin(np.arange(50000)),
        18000,
        24000,
        10000,
        1000,
    )

    assert type(outcome) is bytes  # nosec


def testing_miettinen_bit_generation_interface():
    import numpy as np

    from protocols.miettinen import Miettinen_Protocol

    outcome = Miettinen_Protocol.miettinen_algo(
        np.sin(np.arange(50000)), 1000, 1000, 0.5, 0.5
    )

    assert type(outcome) is bytes  # nosec


def testing_perceptio_bit_generation_interface():
    import numpy as np

    from protocols.perceptio import Perceptio_Protocol

    rng = np.random.default_rng(0)
    signal = rng.integers(0, 10, size=100000)

    fps, events = Perceptio_Protocol.perceptio(signal, 4, 48000, 0.75, 2, 0.1, 2, 5, 3)

    assert type(fps) is list  # nosec
    assert type(fps[0]) is bytes  # nosec

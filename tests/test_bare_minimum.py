import pytest
import sys
import os
from typing import List
sys.path.insert(1, os.getcwd() + "/src")

def testing_schurmann_bit_generation_interface():
    from src.protocols.shurmann import Shurmann_Siggs_Protocol
    import numpy as np

    outcome = Shurmann_Siggs_Protocol.zero_out_antialias_sigs_algo(
            np.sin(np.arange(50000)),
            18000,
            24000,
            10000,
            1000,
        )
    
    assert type(outcome) == bytes


def testing_miettinen_bit_generation_interface():
    from src.protocols.miettinen import Miettinen_Protocol
    import numpy as np

    outcome = Miettinen_Protocol.miettinen_algo(np.sin(np.arange(50000)), 1000, 1000, 0.5, 0.5)
    
    assert type(outcome) == bytes


def testing_perceptio_bit_generation_interface():
    from src.protocols.perceptio import Perceptio_Protocol
    import numpy as np

    rng = np.random.default_rng(0)
    signal = rng.integers(0, 10, size=100000)

    fps, events = Perceptio_Protocol.perceptio(signal, 4, 48000, 0.75, 2, 0.1, 2, 5, 3)

    assert type(fps) == list
    assert type(fps[0]) == bytes



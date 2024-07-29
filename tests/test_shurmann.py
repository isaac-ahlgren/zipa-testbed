import numpy as np
from typing import List

from src.protocols.shurmann import Shurmann_Siggs_Protocol
from src.protocols.protocol_interface import ProtocolInterface
from src.sensors.test_sensor import TestSensor
from src.sensors.sensor_reader import SensorReader 


TEST_SENSOR: dict[str, int] = {
    "sample_rate": 44100,
    "chunk_size": 1024,
    "time_collected": 3,
}

SHURMANN_TEST: dict[str, int] = {
    "window_len": 10000,
    "band_len": 1000,
    "key_length": 8,
    "parity_symbols": 4,
    "sensor": "TEST_SENSOR",
    "timeout": 10,
    "verbose": True
}

DUMMY_PARAMETERS = {
    "verbose": "True",
    "key_length": 8,
    "parity_symbols": 4,
    "timeout": 10,
}

def test_process_context():
    test_sensor = TestSensor(TEST_SENSOR, signal_type="random")
    test_sensor_reader = SensorReader(test_sensor)
    data_reader = ProtocolInterface(DUMMY_PARAMETERS, test_sensor_reader, None)
    Shurmann_Interface = Shurmann_Siggs_Protocol(SHURMANN_TEST, test_sensor_reader, None)

    data_reader.time_length = Shurmann_Interface.time_length
    test_sensor.antialias_sample_rate = 23000
    test_sensor.samplr_rate = 44100
    
    data = test_sensor.read()
    signal = data_reader.read_samples(data_reader.time_length)
    
    shurmann_data = Shurmann_Siggs_Protocol.zero_out_antialias_sigs_algo(
        signal,
        23000,
        44100,
    )

    process_bits = Shurmann_Interface.process_context()

    print(f'data: {shurmann_data}')
    print(f'bits: {process_bits}')

    assert (process_bits == shurmann_data)
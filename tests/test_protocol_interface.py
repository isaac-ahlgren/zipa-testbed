import os
import sys

sys.path.insert(1, "/src")
sys.path.insert(1, os.getcwd() + "/src/sensors")


from src.protocols.protocol_interface import (  # noqa: E402
    COMPLETE,
    PROCESSING,
    READY,
    ProtocolInterface,
)
from src.sensors.sensor_reader import SensorReader  # noqa: E402
from src.sensors.test_sensor import TestSensor  # noqa: E402

DUMMY_PARAMETERS = {
    "verbose": "True",
    "key_length": 8,
    "parity_symbols": 4,
    "timeout": 10,
}

SEN_TEST: dict[str, int] = {
    "sample_rate": 44100,
    "chunk_size": 1024,
    "time_collected": 3,
}


def test_flag_changes() -> None:
    test_sensor = TestSensor(SEN_TEST, signal_type="sine")
    test_sensor_reader = SensorReader(test_sensor)
    test_interface = ProtocolInterface(DUMMY_PARAMETERS, test_sensor_reader, None)
    # Processing value should be 0 by default
    assert test_interface.processing_flag.value == READY  # nosec

    # Flag is captured to collect and process data
    ProtocolInterface.capture_flag(test_interface.processing_flag)
    assert test_interface.processing_flag.value == PROCESSING  # nosec

    # Flag alerts all other processes that results are ready
    ProtocolInterface.release_flag(test_interface.processing_flag)
    assert test_interface.processing_flag.value == COMPLETE  # nosec

    # No processes are using shared results, can be safely destroyed
    ProtocolInterface.reset_flag(test_interface.processing_flag)
    assert test_interface.processing_flag.value == READY  # nosec


def test_shared_memory() -> None:
    pass


def test_get_context() -> None:
    pass


def test_read_samples() -> None:
    pass


def test_clear_queue() -> None:
    pass

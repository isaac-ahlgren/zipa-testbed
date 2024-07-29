import os
import sys
from multiprocessing.shared_memory import ShareableList
from typing import List

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
from src.protocols.shurmann import Shurmann_Siggs_Protocol

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
    test_sensor = TestSensor(SEN_TEST, signal_type="sine")
    test_sensor_reader = SensorReader(test_sensor)
    test_interface = ProtocolInterface(DUMMY_PARAMETERS, test_sensor_reader, None)

    # Create dummy protocol name and dummy byte string
    test_interface.name = "Test_Interface"
    dummy_byte_list = [bytes([1, 2, 3]), bytes([4, 5, 6])]

    # Send dummy list to shared memory and check if its there
    test_interface.write_shm(dummy_byte_list)
    # Need to get the shared memory's list, not the object itself
    write_shm_shared_list = list(ShareableList(name=test_interface.name + "_Bytes"))
    assert dummy_byte_list == write_shm_shared_list  # nosec

    # Check if dummy list is the same when retreiving using read_shm()
    read_shm_shared_list = test_interface.read_shm()
    assert dummy_byte_list == read_shm_shared_list  # nosec

# TODO: run get_context wtih one process running 
# if it runs get_context, it has the same bits as process_context
def test_get_context() -> None:
    """
    Create two processes, one will process data, the other will standby.
    Got to check for both these cases. Create a quick process_context function
    here.
    """

    def process_context() -> List[bytes]:
        return [bytes([1, 2, 3]), bytes([4, 5, 6])]

    test_sensor = TestSensor(SEN_TEST, signal_type="sine")
    test_sensor_reader = SensorReader(test_sensor)
    pi = ProtocolInterface(DUMMY_PARAMETERS, test_sensor_reader, None)
    # a_protocol = Shurmann_Siggs_Protocol(ProtocolInterface)
    pi.name = "Test_Interface"
    pi.process_context = process_context
    bits = pi.get_context()

    # print(f'process_context: {pi.process_context()}')
    # print(f'get_context: {bits}')

    assert bits == pi.process_context()

# TODO: check to see if the array is not empty
# check to see if the array size is the same size as the chunk
# check to see if chunk data is the same as the output data
def test_read_samples() -> None:
    pass

# TODO: check to see if the queue is empty
def test_clear_queue() -> None:
    pass

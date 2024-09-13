import os
import socket
import sys
from multiprocessing import Process

sys.path.insert(1, os.getcwd() + "/src")

from networking.nfs import NFSLogger  # noqa: E402
from protocols.miettinen import Miettinen_Protocol  # noqa: E402
from sensors.sensor_reader import SensorReader  # noqa: E402
from sensors.test_sensor import TestSensor  # noqa: E402

PROTOCOL_DUMMY_PARAMETERS = {
    "f": 1,
    "w": 1,
    "rel_thresh": 0.002,
    "abs_thresh": 0.001,
    "auth_thresh": 0.9,
    "success_thresh": 10,
    "max_iterations": 1,
    "key_length": 2,
    "parity_symbols": 1,
    "sensor": "TestSensor",
    "timeout": 10,
    "verbose": True,
}

SENSOR_DUMMY_PARAMETERS = {
    "sample_rate": 48_000,
    "time_collected": 400,
    "chunk_size": 1_024,
}

DUMMY_LOGGER = NFSLogger(
    user="",
    password="",
    host="SERVER IP",
    database="file_log",
    nfs_server_dir="./local_data",  # Make sure this directory exists and is writable
    local_dir="./local_data",
    identifier="DEVICE IDENTIFIER",  # Could be IP address or any unique identifier
    use_local_dir=True,
)  # nosec


def test_protocol_interaction():
    print("In test")
    test_sensor = TestSensor(SENSOR_DUMMY_PARAMETERS, signal_type="random")
    test_reader = SensorReader(test_sensor)
    test_protocol = Miettinen_Protocol(
        PROTOCOL_DUMMY_PARAMETERS, test_reader, DUMMY_LOGGER
    )

    print("Creating processes")
    host_process = Process(target=host, args=[test_protocol], name="[HOST]")
    device_process = Process(target=device, args=[test_protocol], name="[CLIENT]")

    print("Starting processes.")
    host_process.start()
    device_process.start()

    print("Joining processes")
    host_process.join()
    device_process.join()
    test_reader.poll_process.terminate()

    assert host_process.exitcode == 0  # nosec
    assert device_process.exitcode == 0  # nosec


def host(protocol):
    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_socket.bind(("127.0.0.1", 2000))
    host_socket.listen()
    connection, _ = host_socket.accept()
    host_socket.setblocking(0)
    protocol.host_protocol_single_threaded(connection)


def device(protocol):
    device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    device_socket.connect(("127.0.0.1", 2000))
    protocol.device_protocol(device_socket)


if __name__ == "__main__":
    test_protocol_interaction()
    exit()

import os
import sys
import socket
from multiprocessing import Process

sys.path.insert(1, os.getcwd() + "/src")

from networking.nfs import NFSLogger  # noqa: E402
from protocols.fastzip import FastZIPProtocol  # noqa: E402
from sensors.test_sensor import TestSensor  # noqa: E402
from sensors.sensor_reader import SensorReader  # noqa: E402

# Define protocol and sensor parameters
FASTZIP_PROTOCOL_PARAMETERS = {
    "verbose": True,
        "chunk_size": 200,
        "n_bits": 18,
        "power_thresh": 5,
        "snr_thresh": 2,
        "peak_thresh": 10,
        "bias": 0,
        "sample_rate": 1000,
        "eqd_delta": 5,
        "peak_status": True,
        "ewma_filter": True,
        "alpha": 0.015,
        "remove_noise": True,
        "normalize": True,
        "key_length": 16,  # Make sure to define all necessary parameters
        "parity_symbols": 2,  # Added this line to include parity_symbols
        "timeout": 10
}

SENSOR_DUMMY_PARAMETERS = {
    "sample_rate": 44100,
    "time_collected": 400,
    "chunk_size": 1024,
}

# Set up a dummy logger
DUMMY_LOGGER = NFSLogger(
    user="",
    password="",
    host="SERVER IP",
    database="file_log",
    nfs_server_dir="./local_data",  # Make sure this directory exists and is writable
    local_dir="./local_data/",
    identifier="DEVICE IDENTIFIER",  # Could be IP address or any unique identifier
    use_local_dir=True,
)

def test_fastzip_interaction():
    """ Test FastZIP by simulating host and device interactions. """
    print("Setting up the test.")
    test_sensor = TestSensor(SENSOR_DUMMY_PARAMETERS, signal_type="random")
    test_reader = SensorReader(test_sensor)
    fastzip_protocol = FastZIPProtocol(FASTZIP_PROTOCOL_PARAMETERS, test_reader, DUMMY_LOGGER)

    # Create multiprocessing processes for host and device
    print("Creating processes")
    host_process = Process(target=host, args=(fastzip_protocol,))
    device_process = Process(target=device, args=(fastzip_protocol,))

    # Start both processes
    print("Starting the host and device simulation.")
    host_process.start()
    device_process.start()

    # Wait for both to finish
    host_process.join()
    device_process.join()
    print("Test completed.")

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
    test_fastzip_interaction()

import os
import socket
import sys
import pytest
from multiprocessing import Process
from cryptography.hazmat.primitives.asymmetric import ec

# Assuming your PartitionedGPAKE is in the path
sys.path.insert(1, os.getcwd() + "/src")

from error_correction.GPAKE import PartitionedGPAKE  # Import your GPAKE implementation
from networking.network import gpake_msg_standby, send_gpake_msg, send_status  # Import network functions

@pytest.fixture
def setup_keys():
    key_length = 32
    timeout = 10
    passwords = [b"password1", b"password2", b"password3"]  # Example passwords
    grouped_events = [
        [(0, 1), (2, 3)],  # Event group 1
        [(4, 5), (6, 7)],  # Event group 2
        [(8, 9), (10, 11)],  # Event group 3
    ]
    return key_length, timeout, passwords, grouped_events


def test_network_communication():
    d1 = os.urandom(7)
    d2 = os.urandom(3)
    d3 = os.urandom(4)

    def device():
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2000))
        send_gpake_msg(device_socket, [d1, d2, d3])

    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_socket.bind(("127.0.0.1", 2000))
    host_socket.listen()
    connection, _ = host_socket.accept()
    host_socket.setblocking(0)

    msg = gpake_msg_standby(connection, 10)

    recv_d1 = msg[0]
    recv_d2 = msg[1]
    recv_d3 = msg[2]

    host_socket.close()
    device_process.join()
    assert d1 == recv_d1  # nosec
    assert d2 == recv_d2  # nosec
    assert d3 == recv_d3  # nosec


def test_partitioned_gpake_host_protocol(setup_keys):
    key_length, timeout, passwords, grouped_events = setup_keys
    gpake = PartitionedGPAKE(key_length, timeout)

    def device():
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2000))
        sk1 = gpake.device_protocol(passwords, grouped_events, device_socket, "device", ["host"])

    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_socket.bind(("127.0.0.1", 2000))
    host_socket.listen()
    connection, _ = host_socket.accept()
    host_socket.setblocking(0)

    sk2 = gpake.host_protocol(passwords, grouped_events, connection)
    device_process.join()

    assert sk2 is not None  # nosec


def test_partitioned_gpake_device_protocol(setup_keys):
    key_length, timeout, passwords, grouped_events = setup_keys
    gpake = PartitionedGPAKE(key_length, timeout)

    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 2000))
        host_socket.listen()
        connection, _ = host_socket.accept()
        host_socket.setblocking(0)
        sk2 = gpake.host_protocol(passwords, grouped_events, connection)

    host_process = Process(target=host, name="[SERVER]")
    host_process.start()

    device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    device_socket.connect(("127.0.0.1", 2000))
    sk1 = gpake.device_protocol(passwords, grouped_events, device_socket, "device", ["host"])

    host_process.join()

    assert sk1 is not None  # nosec


if __name__ == "__main__":
    pytest.main([__file__])


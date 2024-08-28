import os
import sys
from multiprocessing import Process
import socket
import time
import numpy as np

sys.path.insert(1, os.getcwd() + "/src")

from networking.network import (
    send_status,
    status_standby,
    ack,
    ack_standby,
    send_commit,
    commit_standby,
    dh_exchange,
    dh_exchange_standby,
    send_nonce_msg,
    get_nonce_msg_standby
)

def check_timing(start_time: float, end_time: float) -> None:
    duration = end_time - start_time
    np.testing.assert_almost_equal(duration, 5.0, decimal=1, err_msg=f"Timing failed: Duration was {duration:.2f} seconds, expected 5.0 seconds.")

def test_send_status():
    start_time = time.time()

    def device(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as device_socket:
            device_socket.connect(("127.0.0.1", port))
            send_status(device_socket, True)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as host_socket:
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 0))  # Bind to an available port
        port = host_socket.getsockname()[1]
        host_socket.listen(1)

        device_process = Process(target=device, args=(port,), name="[CLIENT]")
        device_process.start()

        connection, _ = host_socket.accept()
        test = status_standby(connection, 5)
        success = (test is not None)

    device_process.join()
    end_time = time.time()
    check_timing(start_time, end_time)
    assert success == True, "test_send_status failed"
    time.sleep(1)  # Delay before the next test

def test_ack():
    start_time = time.time()

    def device(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as device_socket:
            device_socket.connect(("127.0.0.1", port))
            ack(device_socket)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as host_socket:
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 0))  # Bind to an available port
        port = host_socket.getsockname()[1]
        host_socket.listen(1)

        device_process = Process(target=device, args=(port,), name="[CLIENT]")
        device_process.start()

        connection, _ = host_socket.accept()
        test = ack_standby(connection, 5)
        success = (test is not None)

    device_process.join()
    end_time = time.time()
    check_timing(start_time, end_time)
    assert success, "test_ack failed"
    time.sleep(1)  # Delay before the next test

def test_send_commit():
    start_time = time.time()

    def device(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as device_socket:
            device_socket.connect(("127.0.0.1", port))
            commitments_list = [b'\x9f\x83\x5d\x7d\x3c\x23\x44\x55', b'\x1b\x9a\x2d\xef\x38\x4f\x63\x9c', b'\x4a\x62\x7d\xd3\x54\xc9\x8b\x03']
            hashes_list = [b'\x6a\x8f\x44\xee\x21\x78\x3b\x97', b'\x1d\x9b\x7c\x5a\x2e\xaf\x76\x84', b'\x38\x5e\x16\x2a\x68\x7c\x9d\x51']
            send_commit(commitments_list, hashes_list, device_socket)

    commitments_list = [b'\x9f\x83\x5d\x7d\x3c\x23\x44\x55', b'\x1b\x9a\x2d\xef\x38\x4f\x63\x9c', b'\x4a\x62\x7d\xd3\x54\xc9\x8b\x03']
    hashes_list = [b'\x6a\x8f\x44\xee\x21\x78\x3b\x97', b'\x1d\x9b\x7c\x5a\x2e\xaf\x76\x84', b'\x38\x5e\x16\x2a\x68\x7c\x9d\x51']

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as host_socket:
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 0))  # Bind to an available port
        port = host_socket.getsockname()[1]
        host_socket.listen(1)

        device_process = Process(target=device, args=(port,), name="[CLIENT]")
        device_process.start()

        connection, _ = host_socket.accept()
        commitments, hashes = commit_standby(connection, 5)
        success = (commitments == commitments_list and hashes == hashes_list)

    device_process.join()
    end_time = time.time()
    check_timing(start_time, end_time)
    assert success, "test_send_commit failed"
    time.sleep(1)  # Delay before the next test

def test_dh_exchange():
    start_time = time.time()

    def device(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as device_socket:
            device_socket.connect(("127.0.0.1", port))
            dh_exchange(device_socket, b'diffiehellmankey')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as host_socket:
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 0))  # Bind to an available port
        port = host_socket.getsockname()[1]
        host_socket.listen(1)

        device_process = Process(target=device, args=(port,), name="[CLIENT]")
        device_process.start()

        connection, _ = host_socket.accept()
        key = dh_exchange_standby(connection, 5)
        success = (key == b'diffiehellmankey')

    device_process.join()
    end_time = time.time()
    check_timing(start_time, end_time)
    assert success, "test_dh_exchange failed"
    time.sleep(1)  # Delay before the next test

def test_send_nonce_msg():
    start_time = time.time()

    def device(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as device_socket:
            device_socket.connect(("127.0.0.1", port))
            send_nonce_msg(device_socket, b'nonce')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as host_socket:
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 0))  # Bind to an available port
        port = host_socket.getsockname()[1]
        host_socket.listen(1)

        device_process = Process(target=device, args=(port,), name="[CLIENT]")
        device_process.start()

        connection, _ = host_socket.accept()
        nonce = get_nonce_msg_standby(connection, 5)
        success = (nonce == b'nonce')

    device_process.join()
    end_time = time.time()
    check_timing(start_time, end_time)
    assert success, "test_send_nonce_msg failed"
    time.sleep(1)  # Delay before the next test

def test_status_standby_timeout():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as host_socket:
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 2000))
        host_socket.listen(1)
        connection, _ = host_socket.accept()
         
        start_time = time.time()
        test = status_standby(connection, 10)  # Open for 10 seconds
        end_time = time.time()
        np.testing.assert_almost_equal(end_time - start_time, 10.0, decimal=1, err_msg="Status standby did not remain open for 10 seconds.")
        assert test is None, "test_status_standby_timeout failed"

if __name__ == '__main__':
    test_send_status()
    test_ack()
    test_send_commit()
    test_dh_exchange()
    test_send_nonce_msg()
    test_status_standby_timeout()

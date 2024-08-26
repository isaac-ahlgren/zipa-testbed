import os
import sys
from multiprocessing import Process
import socket
import time

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

def log_time(test_name: str, start_time: float, end_time: float, success: bool) -> None:
    duration = end_time - start_time
    status = "PASS" if success else "FAIL"
    print(f"{test_name} started at {start_time:.1f} seconds")
    print(f"{test_name} stopped at {end_time:.1f} seconds")
    print(f"Total duration: {duration:.1f} seconds")
    print(f"Test result: {status}")
    print()

def test_send_status():
    start_time = time.time()

    def device():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as device_socket:
            device_socket.connect(("127.0.0.1", 2000))
            send_status(device_socket, True)

    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as host_socket:
        host_socket.bind(("127.0.0.1", 2000))
        host_socket.listen(1)
        connection, _ = host_socket.accept()

        test = status_standby(connection, 5)
        success = (test == True)

    device_process.join()
    end_time = time.time()
    log_time("test_send_status", start_time, end_time, success)

def test_ack():
    start_time = time.time()

    def device():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as device_socket:
            device_socket.connect(("127.0.0.1", 2001))
            ack(device_socket)

    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as host_socket:
        host_socket.bind(("127.0.0.1", 2001))
        host_socket.listen(1)
        connection, _ = host_socket.accept()

        test = ack_standby(connection, 5)
        success = (test == True)

    device_process.join()
    end_time = time.time()
    log_time("test_ack", start_time, end_time, success)

def test_send_commit():
    start_time = time.time()

    def device():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as device_socket:
            device_socket.connect(("127.0.0.1", 2050))
            commitments_list = [b'\x9f\x83\x5d\x7d\x3c\x23\x44\x55', b'\x1b\x9a\x2d\xef\x38\x4f\x63\x9c', b'\x4a\x62\x7d\xd3\x54\xc9\x8b\x03']
            hashes_list = [b'\x6a\x8f\x44\xee\x21\x78\x3b\x97', b'\x1d\x9b\x7c\x5a\x2e\xaf\x76\x84', b'\x38\x5e\x16\x2a\x68\x7c\x9d\x51']
            send_commit(commitments_list, hashes_list, device_socket)

    commitments_list = [b'\x9f\x83\x5d\x7d\x3c\x23\x44\x55', b'\x1b\x9a\x2d\xef\x38\x4f\x63\x9c', b'\x4a\x62\x7d\xd3\x54\xc9\x8b\x03']
    hashes_list = [b'\x6a\x8f\x44\xee\x21\x78\x3b\x97', b'\x1d\x9b\x7c\x5a\x2e\xaf\x76\x84', b'\x38\x5e\x16\x2a\x68\x7c\x9d\x51']

    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as host_socket:
        host_socket.bind(("127.0.0.1", 2050))
        host_socket.listen(1)
        connection, _ = host_socket.accept()

        commitments, hashes = commit_standby(connection, 5)
        success = (commitments == commitments_list and hashes == hashes_list)

    device_process.join()
    end_time = time.time()
    log_time("test_send_commit", start_time, end_time, success)

def test_dh_exchange():
    start_time = time.time()

    def device():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as device_socket:
            device_socket.connect(("127.0.0.1", 2003))
            dh_exchange(device_socket, b'diffiehellmankey')

    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as host_socket:
        host_socket.bind(("127.0.0.1", 2003))
        host_socket.listen(1)
        connection, _ = host_socket.accept()

        key = dh_exchange_standby(connection, 5)
        success = (key == b'diffiehellmankey')

    device_process.join()
    end_time = time.time()
    log_time("test_dh_exchange", start_time, end_time, success)

def test_send_nonce_msg():
    start_time = time.time()

    def device():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as device_socket:
            device_socket.connect(("127.0.0.1", 2004))
            send_nonce_msg(device_socket, b'nonce')

    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as host_socket:
        host_socket.bind(("127.0.0.1", 2004))
        host_socket.listen(1)
        connection, _ = host_socket.accept()

        nonce = get_nonce_msg_standby(connection, 5)
        success = (nonce == b'nonce')

    device_process.join()
    end_time = time.time()
    log_time("test_send_nonce_msg", start_time, end_time, success)

if __name__ == '__main__':
    test_send_status()
    test_ack()
    test_send_commit()
    test_dh_exchange()
    test_send_nonce_msg()


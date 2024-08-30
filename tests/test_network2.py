import os
import socket
import sys
from multiprocessing import Process

sys.path.insert(1, os.getcwd() + "/src")

from networking.network import (  # noqa
    SUCC,
    ack,
    ack_standby,
    commit_standby,
    dh_exchange,
    dh_exchange_standby,
    get_nonce_msg_standby,
    send_commit,
    send_nonce_msg,
    send_status,
    status_standby,
)


def test_send_status():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 2000))
        host_socket.listen()
        connection, _ = host_socket.accept()
        host_socket.setblocking(0)
        send_status(connection, True)

    def device():
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2000))
        result = device_socket.recv(8)
        print(f"Client recieved: {result}")
        # result = status_standby(device_socket, 10)
        assert result == SUCC.encode()  # nosec

    print("Testing send status.\nCreating processes.")
    host_process = Process(target=host, args=[], name="[HOST]")
    client_process = Process(target=device, args=[], name="[CLIENT]")

    print("Starting processes.")
    host_process.start()
    client_process.start()

    print("Joining processes.")
    host_process.join()
    client_process.join()

    assert host_process.exitcode == 0  # nosec
    assert client_process.exitcode == 0  # nosec


def test_status_standby():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 2000))
        host_socket.listen()
        connection, _ = host_socket.accept()
        host_socket.setblocking(0)
        connection.send(SUCC.encode())

    def device():
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2000))
        result = status_standby(device_socket, 10)
        print(f"Client recieved: {result}")
        # result = status_standby(device_socket, 10)
        assert result  # nosec

    print("Testing send status.\nCreating processes.")
    host_process = Process(target=host, args=[], name="[HOST]")
    client_process = Process(target=device, args=[], name="[CLIENT]")

    print("Starting processes.")
    host_process.start()
    client_process.start()

    print("Joining processes.")
    host_process.join()
    client_process.join()

    assert host_process.exitcode == 0  # nosec
    assert client_process.exitcode == 0  # nosec

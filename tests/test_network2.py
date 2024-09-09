import os
import socket
import sys
from multiprocessing import Process
import time

sys.path.insert(1, os.getcwd() + "/src")


from networking.network import (  # noqa
    SUCC,
    ACKN,
    DHKY,
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

def test_ack():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 2000))
        host_socket.listen()
        connection, _ = host_socket.accept()
        ack(connection)  # Send the acknowledgment message

    def device():
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2000))
        result = device_socket.recv(8)  # Adjust buffer size if needed
        print(f"Client received: {result}")
        assert result == ACKN.encode()  # Verify the received message

    print("Testing ack function.\nCreating processes.")
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

    
 
def test_send_commit():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 2050))
        host_socket.listen()
        connection, _ = host_socket.accept()
        commitments, hashes = commit_standby(connection, 5)
        print(f"Host received commitments: {commitments}")
        print(f"Host received hashes: {hashes}")
        connection.close()  # Ensure the connection is closed after use

    def device():
        time.sleep(1)  # Ensure the host is ready before connecting
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2050))

        commitments_list = [b'\x9f\x83\x5d\x7d\x3c\x23\x44\x55', b'\x1b\x9a\x2d\xef\x38\x4f\x63\x9c', b'\x4a\x62\x7d\xd3\x54\xc9\x8b\x03']
        hashes_list = [b'\x6a\x8f\x44\xee\x21\x78\x3b\x97', b'\x1d\x9b\x7c\x5a\x2e\xaf\x76\x84', b'\x38\x5e\x16\x2a\x68\x7c\x9d\x51']

        send_commit(commitments_list, hashes_list, device_socket)
        device_socket.close()  # Ensure the connection is closed after use

    print("Testing send_commit function.\nCreating processes.")
    host_process = Process(target=host, name="[HOST]")
    device_process = Process(target=device, name="[CLIENT]")

    print("Starting processes.")
    host_process.start()
    device_process.start()

    print("Joining processes.")
    host_process.join()
    device_process.join()

    assert host_process.exitcode == 0
    assert device_process.exitcode == 0




def test_commit_standby():

    commitments_list = [b'\x9f\x83\x5d\x7d\x3c\x23\x44\x55', b'\x1b\x9a\x2d\xef\x38\x4f\x63\x9c']
    hashes_list = [b'\x6a\x8f\x44\xee\x21\x78\x3b\x97', b'\x1d\x9b\x7c\x5a\x2e\xaf\x76\x84']

    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 2050))
        host_socket.listen()
        connection, _ = host_socket.accept()
        host_socket.setblocking(0)

        # Sample data to be sent
       
        send_commit(commitments_list, hashes_list, connection)

    def device():
        time.sleep(1)  # Ensure the host is ready

        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2050))
        commitments, hashes = commit_standby(device_socket, 10)
        print(f"Client received commitments: {commitments}")
        print(f"Client received hashes: {hashes}")

        assert commitments is not None
        assert hashes is not None
        assert len(commitments) == 2
        assert len(hashes) == 2
        assert commitments == commitments_list
        assert hashes == hashes_list
    print("Testing commit_standby function.\nCreating processes.")
    host_process = Process(target=host, name="[HOST]")
    device_process = Process(target=device, name="[DEVICE]")

    print("Starting processes.")
    host_process.start()
    device_process.start()

    print("Joining processes.")
    host_process.join()
    device_process.join()

    assert host_process.exitcode == 0
    assert device_process.exitcode == 0


def test_dh_exchange():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 3000))
        host_socket.listen(1)
        connection, _ = host_socket.accept()

        # Receive and validate the DH key
        message = connection.recv(1024)
        assert message[:8] == DHKY.encode()  # Validate prefix
        key_size = int.from_bytes(message[4:8], byteorder="big")
        key = message[8:8 + key_size]
        print(f"Host received key: {key}")

    def device():
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Introduce a slight delay to ensure the host is ready
        time.sleep(1)

        device_socket.connect(("127.0.0.1", 3000))
        key = b'\x01\x02\x03\x04\x05\x06\x07\x08'
        dh_exchange(device_socket, key)

    print("Testing DH exchange.\nCreating processes.")
    host_process = Process(target=host, args=[], name="[HOST]")
    client_process = Process(target=device, args=[], name="[DEVICE]")

    print("Starting processes.")
    host_process.start()
    client_process.start()

    print("Joining processes.")
    host_process.join()
    client_process.join()

    assert host_process.exitcode == 0  # nosec
    assert client_process.exitcode == 0  # nosec
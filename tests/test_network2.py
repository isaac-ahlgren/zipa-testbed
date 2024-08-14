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

# Test send_status
def test_send_status():
    def device():
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2000))
        send_status(device_socket, True)
        device_socket.close()

    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_socket.bind(("127.0.0.1", 2000))
    host_socket.listen(1)
    connection, _ = host_socket.accept()

    test = status_standby(connection, 5)
    assert test == True

    connection.close()
    host_socket.close()
    device_process.join()

# Test ack
def test_ack():
    def device():
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2001))
        ack(device_socket)
        device_socket.close()

    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_socket.bind(("127.0.0.1", 2001))
    host_socket.listen(1)
    connection, _ = host_socket.accept()

    test = ack_standby(connection, 5)
    assert test == True

    connection.close()
    host_socket.close()
    device_process.join()

# Test send_commit
def test_send_commit():
    def device():
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2050))

        commitments_list = [b'\x9f\x83\x5d\x7d\x3c\x23\x44\x55', b'\x1b\x9a\x2d\xef\x38\x4f\x63\x9c', b'\x4a\x62\x7d\xd3\x54\xc9\x8b\x03']
        hashes_list = [b'\x6a\x8f\x44\xee\x21\x78\x3b\x97', b'\x1d\x9b\x7c\x5a\x2e\xaf\x76\x84', b'\x38\x5e\x16\x2a\x68\x7c\x9d\x51']

        send_commit(commitments_list, hashes_list, device_socket)
        device_socket.close()

    commitments_list = [b'\x9f\x83\x5d\x7d\x3c\x23\x44\x55', b'\x1b\x9a\x2d\xef\x38\x4f\x63\x9c', b'\x4a\x62\x7d\xd3\x54\xc9\x8b\x03']
    hashes_list = [b'\x6a\x8f\x44\xee\x21\x78\x3b\x97', b'\x1d\x9b\x7c\x5a\x2e\xaf\x76\x84', b'\x38\x5e\x16\x2a\x68\x7c\x9d\x51']

    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_socket.bind(("127.0.0.1", 2050))
    host_socket.listen(1)
    connection, _ = host_socket.accept()

    commitments, hashes = commit_standby(connection, 5)

    # print(f'{type(commitments_list)}: {commitments_list}')
    # print(f'{type(commitments)}: {commitments}')
    # print(f'{type(hashes)}: {hashes}')
    # print(f'{type(hashes_list)}: {hashes_list}')

    assert commitments == commitments_list
    assert hashes == hashes_list

    connection.close()
    host_socket.close()
    device_process.join()

# Test dh_exchange
def test_dh_exchange():
    def device():
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2003))
        dh_exchange(device_socket, b'diffiehellmankey')
        device_socket.close()

    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_socket.bind(("127.0.0.1", 2003))
    host_socket.listen(1)
    connection, _ = host_socket.accept()

    key = dh_exchange_standby(connection, 5)
    assert key == b'diffiehellmankey'

    connection.close()
    host_socket.close()
    device_process.join()

# Test send_nonce_msg
def test_send_nonce_msg():
    def device():
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2004))
        send_nonce_msg(device_socket, b'nonce')
        device_socket.close()

    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_socket.bind(("127.0.0.1", 2004))
    host_socket.listen(1)
    connection, _ = host_socket.accept()

    nonce = get_nonce_msg_standby(connection, 5)
    assert nonce == b'nonce'

    connection.close()
    host_socket.close()
    device_process.join()



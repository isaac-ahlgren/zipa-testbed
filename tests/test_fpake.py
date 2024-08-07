import os
import sys
import socket
from multiprocessing import Process, Pipe

sys.path.insert(1, os.getcwd() + "/src/error_correction")

from fPAKE import fPAKE


def test_fpake():
    key = os.urandom(8)
    fpake = fPAKE(10)

    sock1, sock2 = Pipe()

    def device():
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2000))
        sk1 = fpake.device_protocol(key, device_socket)
        sock1.send_bytes(sk1)

    key_ba = bytearray(key)
    key_ba[0] = 0
    other_key = key_ba

    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_socket.bind(("127.0.0.1", 2000))
    host_socket.listen()
    connection, _ = host_socket.accept()
    host_socket.setblocking(0)

    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    sk2 = fpake.host_protocol(key, connection)
    sk1 = sock2.recv_bytes()

    assert sk1 == sk2

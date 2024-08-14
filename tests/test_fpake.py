import os
import sys
import socket
from multiprocessing import Process, Pipe

sys.path.insert(1, os.getcwd() + "/src")

from error_correction.fPAKE import fPAKE
from error_correction.simple_reed_solomon import SimpleReedSolomonObj
from networking.network import send_fpake_msg, fpake_msg_standby

def test_network_communication():
    d1 = os.urandom(7)
    d2 = os.urandom(3)
    d3 = os.urandom(4)

    def device():
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2000))
        send_fpake_msg(device_socket, [d1, d2, d3])

    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_socket.bind(("127.0.0.1", 2000))
    host_socket.listen()
    connection, _ = host_socket.accept()
    host_socket.setblocking(0)
    
    msg = fpake_msg_standby(connection, 10)

    recv_d1 = msg[0]
    recv_d2 = msg[1]
    recv_d3 = msg[2]
    
    host_socket.close()
    device_process.join()
    assert d1 == recv_d1
    assert d2 == recv_d2
    assert d3 == recv_d3

def test_simple_reed_solomon_gf_2_8():
    rs = SimpleReedSolomonObj(8, 6, power_of_2=8, generator=2, prime_poly=0x11D)
    key = b'\x01' +  b'\x01' + os.urandom(4)

    C = rs.encode(key)

    C[0] = 0

    decoded_key = rs.decode(C)

    assert decoded_key == key

    C[1] = 0

    decoded_key = rs.decode(C)

    assert decoded_key != key

def test_simple_reed_solomon_gf_2_12():
    prime_poly = 0x1053
    generator = 2
    rs = SimpleReedSolomonObj(10, 8, power_of_2=16, generator=generator, prime_poly=prime_poly)
    key = os.urandom(12)

    C = rs.encode(key)

    C[0] = 0

    decoded_key = rs.decode(C)

    print(key)
    print(decoded_key)

    assert decoded_key == key

    C[33] = 0

    decoded_key = rs.decode(C)

    assert decoded_key != key

def test_fpake():
    key = b'\x01' + os.urandom(3)
    fpake = fPAKE(4, 2, 10, "..")

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

    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_socket.bind(("127.0.0.1", 2000))
    host_socket.listen()
    connection, _ = host_socket.accept()
    host_socket.setblocking(0)
    
    sk2 = fpake.host_protocol(other_key, connection)
    sk1 = sock2.recv_bytes()
    device_process.join()
    assert sk1 == sk2

if __name__ == "__main__":
    test_simple_reed_solomon_gf_2_16()
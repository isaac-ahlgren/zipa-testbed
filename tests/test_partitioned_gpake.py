import os
import socket
import sys
from multiprocessing import Process, Queue

sys.path.insert(1, os.getcwd() + "/src")

from error_correction.part_gPAKE import GPAKE  # Assuming GPAKE is your GPAKE implementation
from networking.network import pake_msg_standby, send_pake_msg  # Networking methods


def test_gpake_network_communication():
    """Test network communication for GPAKE, ensuring data integrity across devices."""
    d1 = os.urandom(7)
    d2 = os.urandom(3)
    d3 = os.urandom(4)

    def device():
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2000))
        send_pake_msg(device_socket, [d1, d2, d3])

    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_socket.bind(("127.0.0.1", 2000))
    host_socket.listen()
    connection, _ = host_socket.accept()
    host_socket.setblocking(0)

    msg = pake_msg_standby(connection, 10)

    recv_d1 = msg[0]
    recv_d2 = msg[1]
    recv_d3 = msg[2]

    host_socket.close()
    device_process.join()
    
    assert d1 == recv_d1  # nosec
    assert d2 == recv_d2  # nosec
    assert d3 == recv_d3  # nosec


def test_gpake():
    """Test the GPAKE protocol by establishing a shared key between a host and a device."""
    grouped_events = [[1, 2, 3], [4, 5, 6]]  # Example grouped events
    passwords = [b"\x01" * 16, b"\x02" * 16]  # Example 16-byte passwords for each event group

    gpake = GPAKE()  # Initialize the GPAKE protocol
    queue = Queue()  # Queue to store the final device key

    def device():
        device_id = "device_1"  # Unique identifier for the device
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2000))

        # Execute GPAKE device protocol
        final_key_device = gpake.device_protocol(grouped_events, passwords, device_socket, device_id)
        queue.put(final_key_device)  # Put the device key in the queue
        print(f"Final key (device): {final_key_device.hex()}")
        device_socket.close()

    # Start device process
    device_process = Process(target=device, name="[CLIENT]")
    device_process.start()

    host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    host_socket.bind(("127.0.0.1", 2000))
    host_socket.listen()
    connection, _ = host_socket.accept()

    # Execute GPAKE host protocol
    device_id = "host_device"  # Unique identifier for the host
    final_key_host = gpake.host_protocol(grouped_events, passwords, connection)
    print(f"Final key (host): {final_key_host.hex()}")

    # Get final key from the device process
    final_key_device = queue.get()  # Retrieve the device key from the queue

    # Wait for the device to finish
    device_process.join()

    # Ensure both host and device derived the same key
    assert final_key_host == final_key_device  # nosec
    host_socket.close()


if __name__ == "__main__":
    test_gpake_network_communication()  # Test network communication
    test_gpake()  # Test the full GPAKE protocol

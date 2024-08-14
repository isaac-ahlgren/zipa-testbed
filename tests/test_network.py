import os
import socket
import time  # Import the time module
import unittest
from multiprocessing import Process
from typing import List, Optional, Tuple  # Union

# Function to send a message from the device
def send_fpake_msg(connection: socket.socket, data_list):
    """
    Sends a fake message over a network connection.

    :param connection: The network connection to send data over.
    :param data_list: A list of byte sequences to send.
    """
    # Send each piece of data prefixed with its length
    message = b"".join(len(data).to_bytes(4, byteorder="big") + data for data in data_list)
    connection.send(message)

# Function to receive a message on the host
def fpake_msg_standby(connection: socket.socket, timeout: int):
    """
    Waits to receive a fake message within a specified timeout period.

    :param connection: The network connection to receive from.
    :param timeout: The maximum time in seconds to wait for the message.
    :returns: A list of received byte sequences.
    """
    reference = time.time()
    timestamp = reference
    data_list = []

    connection.settimeout(timeout)

    try:
        while True:
            length_bytes = connection.recv(4)
            if not length_bytes:
                break
            length = int.from_bytes(length_bytes, byteorder="big")
            data = connection.recv(length)
            data_list.append(data)
    except socket.timeout:
        pass

    return data_list

class TestNetworkCommunication(unittest.TestCase):

    def test_network_communication(self):
        # Randomly generate test data
        d1 = os.urandom(7)
        d2 = os.urandom(3)
        d3 = os.urandom(4)

        def device():
            device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            device_socket.connect(("127.0.0.1", 2000))
            send_fpake_msg(device_socket, [d1, d2, d3])
            device_socket.close()

        # Start the device process
        device_process = Process(target=device, name="[CLIENT]")
        device_process.start()

        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 2000))
        host_socket.listen(1)
        connection, _ = host_socket.accept()

        # Receive the message
        msg = fpake_msg_standby(connection, 10)

        # Assert that received data matches sent data
        recv_d1 = msg[0]
        recv_d2 = msg[1]
        recv_d3 = msg[2]

        host_socket.close()
        device_process.join()

        self.assertEqual(d1, recv_d1)
        self.assertEqual(d2, recv_d2)
        self.assertEqual(d3, recv_d3)

if __name__ == "__main__":
    unittest.main()




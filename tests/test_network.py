import os
import socket
import sys
import time
from multiprocessing import Process

sys.path.insert(1, os.getcwd() + "/src")


from networking.network import (  # noqa
    ACKN,
    DHKY,
    FPFM,
    NONC,
    SUCC,
    ack,
    ack_standby,
    commit_standby,
    dh_exchange,
    dh_exchange_standby,
    get_nonce_msg_standby,
    get_nonce_msg_standby2,
    pake_msg_standby,
    send_commit,
    send_nonce_msg,
    send_pake_msg,
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
        assert result == ACKN.encode()  # nosec

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

        commitments_list = [
            b"\x9f\x83\x5d\x7d\x3c\x23\x44\x55",
            b"\x1b\x9a\x2d\xef\x38\x4f\x63\x9c",
            b"\x4a\x62\x7d\xd3\x54\xc9\x8b\x03",
        ]
        hashes_list = [
            b"\x6a\x8f\x44\xee\x21\x78\x3b\x97",
            b"\x1d\x9b\x7c\x5a\x2e\xaf\x76\x84",
            b"\x38\x5e\x16\x2a\x68\x7c\x9d\x51",
        ]

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

    assert host_process.exitcode == 0  # nosec
    assert device_process.exitcode == 0  # nosec


def test_commit_standby():

    commitments_list = [
        b"\x9f\x83\x5d\x7d\x3c\x23\x44\x55",
        b"\x1b\x9a\x2d\xef\x38\x4f\x63\x9c",
    ]
    hashes_list = [
        b"\x6a\x8f\x44\xee\x21\x78\x3b\x97",
        b"\x1d\x9b\x7c\x5a\x2e\xaf\x76\x84",
    ]

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

        assert commitments is not None  # nosec
        assert hashes is not None  # nosec
        assert len(commitments) == 2  # nosec
        assert len(hashes) == 2  # nosec
        assert commitments == commitments_list  # nosec
        assert hashes == hashes_list  # nosec

    print("Testing commit_standby function.\nCreating processes.")
    host_process = Process(target=host, name="[HOST]")
    device_process = Process(target=device, name="[DEVICE]")

    print("Starting processes.")
    host_process.start()
    device_process.start()

    print("Joining processes.")
    host_process.join()
    device_process.join()

    assert host_process.exitcode == 0  # nosec
    assert device_process.exitcode == 0  # nosec


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
        assert message[:8] == DHKY.encode()  # nosec
        key_size = int.from_bytes(message[8:12], byteorder="big")
        key = message[12 : 12 + key_size]
        print(f"Host received key: {key}")

    def device():
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Introduce a slight delay to ensure the host is ready
        time.sleep(1)

        device_socket.connect(("127.0.0.1", 3000))
        key = b"\x01\x02\x03\x04\x05\x06\x07\x08"
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


def test_dh_exchange_standby():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 3001))
        host_socket.listen(1)
        connection, _ = host_socket.accept()

        # Receive and validate the DH key using dh_exchange_standby
        key = dh_exchange_standby(connection, 10)
        assert key == b"\x01\x02\x03\x04\x05\x06\x07\x08"  # nosec
        print(f"Host received key: {key}")

    def device():
        time.sleep(1)  # Ensure the host is ready

        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 3001))

        key = b"\x01\x02\x03\x04\x05\x06\x07\x08"
        dh_exchange(device_socket, key)

    print("Testing dh_exchange_standby function.\nCreating processes.")
    host_process = Process(target=host, args=[], name="[HOST]")
    device_process = Process(target=device, args=[], name="[DEVICE]")

    print("Starting processes.")
    host_process.start()
    device_process.start()

    print("Joining processes.")
    host_process.join()
    device_process.join()

    assert host_process.exitcode == 0  # nosec
    assert device_process.exitcode == 0  # nosec


def test_send_nonce_msg():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 4000))
        host_socket.listen()
        connection, _ = host_socket.accept()

        # Expect to receive the nonce message
        message = connection.recv(1024)
        print(f"Host received message: {message}")
        assert message.startswith(NONC.encode())  # nosec

        nonce_size = int.from_bytes(message[8:12], byteorder="big")
        nonce = message[12 : 12 + nonce_size]
        print(f"Host received nonce: {nonce}")

    def device():
        time.sleep(1)  # Ensure the host is ready

        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 4000))

        nonce = b"\x12\x34\x56\x78"
        send_nonce_msg(device_socket, nonce)

    print("Testing send_nonce_msg function.\nCreating processes.")
    host_process = Process(target=host, args=[], name="[HOST]")
    device_process = Process(target=device, args=[], name="[DEVICE]")

    print("Starting processes.")
    host_process.start()
    device_process.start()

    print("Joining processes.")
    host_process.join()
    device_process.join()

    assert host_process.exitcode == 0  # nosec
    assert device_process.exitcode == 0  # nosec


def test_get_nonce_msg_standby2():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 5000))
        host_socket.listen()
        connection, _ = host_socket.accept()

        # Expect to receive the nonce message
        nonce = get_nonce_msg_standby2(connection, 5)
        print(f"Host received nonce: {nonce}")
        assert nonce == b"\x12\x34\x56\x78"  # nosec

    def device():
        time.sleep(1)  # Ensure the host is ready

        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 5000))

        nonce = b"\x12\x34\x56\x78"
        send_nonce_msg(device_socket, nonce)

    print("Testing get_nonce_msg_standby2 function.\nCreating processes.")
    host_process = Process(target=host, args=[], name="[HOST]")
    device_process = Process(target=device, args=[], name="[DEVICE]")

    print("Starting processes.")
    host_process.start()
    device_process.start()

    print("Joining processes.")
    host_process.join()
    device_process.join()

    assert host_process.exitcode == 0  # nosec
    assert device_process.exitcode == 0  # nosec


def test_get_nonce_msg_standby():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 6000))
        host_socket.listen()
        connection, _ = host_socket.accept()

        # Expect to receive the nonce message
        nonce = get_nonce_msg_standby(connection, 5)
        print(f"Host received nonce: {nonce}")
        assert nonce == b"\x12\x34\x56\x78"  # nosec

    def device():
        time.sleep(1)  # Ensure the host is ready

        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 6000))

        nonce = b"\x12\x34\x56\x78"
        send_nonce_msg(device_socket, nonce)

    print("Testing get_nonce_msg_standby function.\nCreating processes.")
    host_process = Process(target=host, args=[], name="[HOST]")
    device_process = Process(target=device, args=[], name="[DEVICE]")

    print("Starting processes.")
    host_process.start()
    device_process.start()

    print("Joining processes.")
    host_process.join()
    device_process.join()

    assert host_process.exitcode == 0  # nosec
    assert device_process.exitcode == 0  # nosec


def test_ack_standby():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 2000))
        host_socket.listen()
        connection, _ = host_socket.accept()
        time.sleep(1)  # Simulate delay before sending ACK
        connection.send(ACKN.encode())  # Send the acknowledgment

    def device():
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2000))

        acknowledged = ack_standby(
            device_socket, 5
        )  # Wait for ACK with a 5-second timeout
        print(f"Client received acknowledgment: {acknowledged}")
        assert acknowledged is True  # nosec

    print("Testing ack_standby function.\nCreating processes.")
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


def test_send_fpake_msg():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 2001))
        host_socket.listen()
        connection, _ = host_socket.accept()

        # Receive the sent message
        sent_data = connection.recv(1024)

        # Expected payload

        length_payload = (len(b"hello") + 4 + len(b"world") + 4).to_bytes(
            4, byteorder="big"
        )
        expected_payload = (
            FPFM.encode()
            + length_payload
            + len(b"hello").to_bytes(4, byteorder="big")
            + b"hello"
            + len(b"world").to_bytes(4, byteorder="big")
            + b"world"
        )

        # Assert the received data matches the expected payload
        assert sent_data == expected_payload  # nosec
        print("Host received correct data.")

    def device():
        time.sleep(1)  # Simulate delay to ensure host is ready
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2001))

        # Define the message
        msg = [b"hello", b"world"]

        # Send the message
        send_pake_msg(device_socket, msg)
        print("Device sent the message.")

    # Creating processes for host and device
    host_process = Process(target=host, args=[], name="[HOST]")
    device_process = Process(target=device, args=[], name="[DEVICE]")

    # Starting processes
    host_process.start()
    device_process.start()

    # Joining processes
    host_process.join()
    device_process.join()

    # Check exit codes of both processes
    assert host_process.exitcode == 0  # nosec
    assert device_process.exitcode == 0  # nosec


def test_fpake_msg_standby():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 2001))
        host_socket.listen()
        connection, _ = host_socket.accept()

        # Send a message
        msg = [b"hello", b"world"]
        send_pake_msg(connection, msg)
        print("Host sent the message.")

    def device():
        time.sleep(1)  # Simulate delay to ensure host is ready
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 2001))

        # Wait for the message using fpake_msg_standby
        received_msg = pake_msg_standby(device_socket, timeout=5)

        # Expected message
        expected_msg = [b"hello", b"world"]

        # Assert the received message matches the expected message
        assert received_msg == expected_msg  # nosec
        print("Device received the correct message.")

    # Creating processes for host and device
    host_process = Process(target=host, args=[], name="[HOST]")
    device_process = Process(target=device, args=[], name="[DEVICE]")

    # Starting processes
    host_process.start()
    device_process.start()

    # Joining processes
    host_process.join()
    device_process.join()

    # Check exit codes of both processes
    assert host_process.exitcode == 0  # nosec
    assert device_process.exitcode == 0  # nosec


def test_status_standby_timeout_early_close():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 8001))
        host_socket.listen()
        connection, _ = host_socket.accept()
        duration = 10
        host_socket.setblocking(False)
        host_socket.settimeout(duration)
        time.sleep(duration)  # Keep the connection open for 5 seconds
        connection.close()  # Close the connection after 5 seconds
        print(f"\nHost connection has been closed after {duration} seconds")
        host_socket.close()

    def device():
        time.sleep(1)  # Ensure the host is ready
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 8001))
        start_time = time.time()
        result = status_standby(device_socket, 5)  # Wait for up to 10 seconds
        end_time = time.time()
        print(f"Status standby lasted {end_time - start_time} seconds.\n")
        device_socket.close()

        # If result is None, it means no data was received and it likely timed out
        assert (  # nosec
            result is None
        ), "Test failed: status_standby should not receive data after early close."

    host_process = Process(target=host, name="[HOST]")
    device_process = Process(target=device, name="[DEVICE]")

    host_process.start()
    device_process.start()

    host_process.join()
    device_process.join()

    assert (  # nosec
        device_process.exitcode == 0
    ), "Test failed: status_standby didn't handle early close correctly."
    print("Test passed: status_standby function handled early close properly.")


def test_commit_standby_timeout_early_close():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 8002))
        host_socket.listen()
        connection, _ = host_socket.accept()
        duration = 10
        host_socket.setblocking(False)
        host_socket.settimeout(duration)
        time.sleep(duration)  # Keep the connection open for 5 seconds
        connection.close()  # Close the connection after 5 seconds
        print(f"\nHost connection has been closed after {duration} seconds")
        host_socket.close()

    def device():
        time.sleep(1)  # Ensure the host is ready
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 8002))
        start_time = time.time()

        # Wait for up to 10 seconds for commit_standby
        commitments, hashes = commit_standby(device_socket, 5)
        end_time = time.time()
        print(f"Commit standby lasted {end_time - start_time} seconds.\n")
        device_socket.close()

        # No commitments or hashes should be received after early close
        assert (  # nosec
            commitments is None and hashes is None
        ), "Test failed: commit_standby should not receive data after early close."

    host_process = Process(target=host, name="[HOST]")
    device_process = Process(target=device, name="[DEVICE]")

    host_process.start()
    device_process.start()

    host_process.join()
    device_process.join()

    assert (  # nosec
        device_process.exitcode == 0
    ), "Test failed: commit_standby didn't handle early close correctly."
    print("Test passed: commit_standby function handled early close properly.")


def test_dh_exchange_standby_timeout_early_close():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 8003))
        host_socket.listen()
        connection, _ = host_socket.accept()
        duration = 10
        host_socket.setblocking(False)
        host_socket.settimeout(duration)
        time.sleep(duration)  # Keep the connection open for 5 seconds
        connection.close()  # Close the connection after 5 seconds
        print(f"\nHost connection has been closed after {duration} seconds")
        host_socket.close()

    def device():
        time.sleep(1)  # Ensure the host is ready
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 8003))
        start_time = time.time()

        # Wait for up to 10 seconds for dh_exchange_standby
        key = dh_exchange_standby(device_socket, 5)
        end_time = time.time()
        print(f"DH exchange standby lasted {end_time - start_time} seconds.\n")
        device_socket.close()

        # No key should be received after early close
        assert (  # nosec
            key is None
        ), "Test failed: dh_exchange_standby should not receive data after early close."

    host_process = Process(target=host, name="[HOST]")
    device_process = Process(target=device, name="[DEVICE]")

    host_process.start()
    device_process.start()

    host_process.join()
    device_process.join()

    assert (  # nosec
        device_process.exitcode == 0
    ), "Test failed: dh_exchange_standby didn't handle early close correctly."
    print("Test passed: dh_exchange_standby function handled early close properly.")


def test_get_nonce_msg_standby_timeout_early_close():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 8004))
        host_socket.listen()
        connection, _ = host_socket.accept()
        duration = 10
        host_socket.setblocking(False)
        host_socket.settimeout(duration)
        time.sleep(duration)  # Keep the connection open for 5 seconds
        connection.close()  # Close the connection after 5 seconds
        print(f"\nHost connection has been closed after {duration} seconds")
        host_socket.close()

    def device():
        time.sleep(1)  # Ensure the host is ready
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 8004))
        start_time = time.time()

        # Wait for up to 10 seconds for get_nonce_msg_standby
        nonce = get_nonce_msg_standby(device_socket, 5)
        end_time = time.time()
        print(f"Get nonce msg standby lasted {end_time - start_time} seconds.\n")
        device_socket.close()

        # No nonce should be received after early close
        assert (  # nosec
            nonce is None
        ), "Test failed: get_nonce_msg_standby should not receive data after early close."

    host_process = Process(target=host, name="[HOST]")
    device_process = Process(target=device, name="[DEVICE]")

    host_process.start()
    device_process.start()

    host_process.join()
    device_process.join()

    assert (  # nosec
        device_process.exitcode == 0
    ), "Test failed: get_nonce_msg_standby didn't handle early close correctly."
    print("Test passed: get_nonce_msg_standby function handled early close properly.")


def test_ack_standby_timeout_early_close():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 8005))
        host_socket.listen()
        connection, _ = host_socket.accept()
        duration = 10
        host_socket.setblocking(False)
        host_socket.settimeout(duration)
        time.sleep(duration)  # Keep the connection open for 5 seconds
        connection.close()  # Close the connection after 5 seconds
        print(f"\nHost connection has been closed after {duration} seconds")
        host_socket.close()

    def device():
        time.sleep(1)  # Ensure the host is ready
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 8005))

        start_time = time.time()

        # Wait for up to 5 seconds for ack_standby
        acknowledged = ack_standby(device_socket, 5)

        end_time = time.time()
        print(f"Device total time: {end_time - start_time} seconds")

        device_socket.close()

        # No acknowledgment should be received after early close
        assert (  # nosec
            acknowledged is False
        ), "Test failed: ack_standby should not receive data after early close."

    host_process = Process(target=host, name="[HOST]")
    device_process = Process(target=device, name="[DEVICE]")

    host_process.start()
    device_process.start()

    host_process.join()
    device_process.join()

    assert (  # nosec
        device_process.exitcode == 0
    ), "Test failed: ack_standby didn't handle early close correctly."
    print("Test passed: ack_standby function handled early close properly.")


def test_fpake_msg_standby_timeout_early_close():
    def host():
        host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host_socket.bind(("127.0.0.1", 8006))
        host_socket.listen()
        connection, _ = host_socket.accept()
        duration = 10
        host_socket.setblocking(False)
        host_socket.settimeout(duration)
        time.sleep(duration)  # Keep the connection open for 5 seconds
        connection.close()  # Close the connection after 5 seconds
        print(f"\nHost connection has been closed after {duration} seconds")
        host_socket.close()

    def device():
        time.sleep(1)  # Ensure the host is ready
        device_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        device_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        device_socket.connect(("127.0.0.1", 8006))
        start_time = time.time()

        # Wait for up to 10 seconds for fpake_msg_standby
        received_msg = pake_msg_standby(device_socket, 5)
        end_time = time.time()
        print(f"FPAKE msg standby lasted {end_time - start_time} seconds.\n")
        device_socket.close()

        # No message should be received after early close
        assert (  # nosec
            received_msg is None
        ), "Test failed: fpake_msg_standby should not receive data after early close."

    host_process = Process(target=host, name="[HOST]")
    device_process = Process(target=device, name="[DEVICE]")

    host_process.start()
    device_process.start()

    host_process.join()
    device_process.join()

    assert (  # nosec
        device_process.exitcode == 0
    ), "Test failed: fpake_msg_standby didn't handle early close correctly."
    print("Test passed: fpake_msg_standby function handled early close properly.")

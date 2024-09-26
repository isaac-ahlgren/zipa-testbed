#clean netowrk
import socket
import time
from typing import List, Optional, Tuple  # Union

# Commands
HOST = "host    "
STRT = "start   "
ACKN = "ack     "
COMM = "comm    "
DHKY = "dhkey   "
NONC = "nonce   "
PRAM = "preamble"
SUCC = "success "
FAIL = "failed  "
FPFM = "fpake   "

def ack(connection: socket.socket) -> None:
    """
    Sends an acknowledgement message over a connection.

    :param connection: The network connection over which to send the acknowledgement.
    """
    connection.send(ACKN.encode())

def ack_standby(connection: socket.socket, timeout: int) -> bool:
    """
    Waits for an acknowledgement within a specified timeout period.

    :param connection: The network connection to receive from.
    :param timeout: The maximum time in seconds to wait for an acknowledgement.
    :returns: True if acknowledged, False if timeout occurred.
    """
    acknowledged = False
    connection.settimeout(timeout)

    try:
        command = connection.recv(8)
    except socket.timeout:
        print("Ack not received within the timeout period.")
        command = None

    if command == ACKN.encode():
        acknowledged = True

    return acknowledged

def commit_standby(
    connection: socket.socket, timeout: int
) -> Tuple[Optional[List[bytes]], Optional[List[bytes]]]:
    """
    Waits for commitments and their hashes to be received within a specified timeout.

    :param connection: The network connection to receive from.
    :param timeout: The maximum time in seconds to wait for the data.
    :returns: A tuple (commitments, hashes) if received, None otherwise.
    """
    reference = time.time()
    timestamp = reference
    commitments = None
    hashes = None

    # Set the socket timeout to ensure recv() does not block indefinitely
    connection.settimeout(timeout)

    try:
        while (timestamp - reference) < timeout:
            timestamp = time.time()
            message = connection.recv(8)

            if message[:8] == COMM.encode():
                # Receive and unpack the lengths for number_of_commits, hash_length, and com_length
                message = connection.recv(12)

                if len(message) != 12:  # Ensure the message is complete
                    print("Incomplete header message received.")
                    continue

                number_of_commits = int.from_bytes(message[0:4], byteorder="big")
                hash_length = int.from_bytes(message[4:8], byteorder="big")
                com_length = int.from_bytes(message[8:12], byteorder="big")

                # Receive the total data based on the calculated size
                commits = connection.recv(
                    number_of_commits * (hash_length + com_length)
                )

                if len(commits) != number_of_commits * (hash_length + com_length):
                    print("Incomplete commit data received.")
                    continue

                # Initialize lists for commitments and hashes
                commitments = []
                hashes = []

                # Extract commitments and hashes from the received data
                for i in range(number_of_commits):
                    commitments.append(
                        commits[
                            i
                            * (hash_length + com_length) : i
                            * (hash_length + com_length)
                            + com_length
                        ]
                    )
                    hashes.append(
                        commits[
                            i * (hash_length + com_length)
                            + com_length : (i + 1) * (hash_length + com_length)
                        ]
                    )
                break  # Exit the loop after successfully receiving the data

    except socket.timeout:
        print("commit not received within the timeout period.")
        return None, None

    return commitments, hashes

def dh_exchange(connection: socket.socket, key: bytes) -> None:
    """
    Sends a Diffie-Hellman key over a network connection.

    :param connection: The network connection to send the key over.
    :param key: The key to be sent.
    """
    key_size = len(key).to_bytes(4, byteorder="big")
    message = DHKY.encode() + key_size + key
    connection.send(message)

def dh_exchange_standby(connection: socket.socket, timeout: int) -> Optional[bytes]:
    """
    Waits to receive a Diffie-Hellman key within a specified timeout period.

    :param connection: The network connection to receive from.
    :param timeout: The maximum time in seconds to wait for the key.
    :returns: The received key if successful, None otherwise.
    """
    reference = time.time()
    key = None

    # Set the socket timeout
    connection.settimeout(timeout)

    try:
        # While process hasn't timed out
        while (time.time() - reference) < timeout:
            # Try to receive the initial message of 12 bytes
            message = connection.recv(12)

            if not message:  # If the connection is closed or no data received
                continue

            # Check if the command matches the expected "DHKY" (assumed to be predefined)
            command = message[:8]
            if command == DHKY.encode():  # Assuming DHKY is a predefined string
                key_size = int.from_bytes(message[8:], "big")

                # Receive the key based on the extracted size
                key = connection.recv(key_size)
                break
    except socket.timeout:
        print("dh not received within the timeout period.")
        return None  # Return None if timeout occurs

    return key

def get_nonce_msg_standby(connection: socket.socket, timeout: int) -> Optional[bytes]:
    """
    Waits to receive a nonce within a specified timeout period.

    :param connection: The network connection to receive from.
    :param timeout: The maximum time in seconds to wait for the nonce.
    :returns: The received nonce if successful, None otherwise.
    """
    nonce = None

    # Set the socket timeout
    connection.settimeout(timeout)

    try:
        # Attempt to receive the initial message of 12 bytes
        message = connection.recv(12)
    except socket.timeout:
        print("Nonce timeout reached.")
        return None  # Return None if timeout occurs
    except TimeoutError:
        print("Unexpected error: TimeoutError occurred.")
        return None

    # If the message is received
    if message:
        command = message[:8]
        if command == NONC.encode():  # Assuming NONC is predefined
            # Extract the size of the nonce from the last 4 bytes
            nonce_size = int.from_bytes(message[8:], "big")

            try:
                # Receive the nonce based on the extracted size
                nonce = connection.recv(nonce_size)
            except socket.timeout:
                print("getnonce not received within the timeout period.")
                return None

    return nonce

def pake_msg_standby(connection: socket.socket, timeout: int) -> bool:

    msg = None
    reference = time.time()
    connection.settimeout(timeout)  # Set socket timeout

    try:
        # While process hasn't timed out
        while (time.time() - reference) < timeout:
            message = connection.recv(12)

            if not message:  # Check for empty or closed connection
                continue

            # Check if message starts with the expected prefix
            if message[:8] == FPFM.encode():
                msg_size = int.from_bytes(message[8:], "big")

                # Receive the payload based on msg_size
                payload = connection.recv(msg_size)

                msg = []
                index = 0
                # Decode the payload by extracting items
                while index < msg_size:
                    item_length = int.from_bytes(payload[index : index + 4], "big")
                    item = payload[index + 4 : index + 4 + item_length]
                    index += 4 + item_length
                    msg.append(item)
                break
    except socket.timeout:
        print("fpake not received within the timeout period.")

    return msg

def send_commit(
    commitments: List[bytes], hashes: List[bytes], device: socket.socket
) -> None:
    """
    Sends commitments along with their corresponding hashes over a network connection.

    :param commitments: List of commitments to be sent.
    :param hashes: List of hashes corresponding to the commitments.
    :param device: The network connection to send data over.
    """
    # Prepare number of commitments and their lengths
    number_of_commitments = len(commitments).to_bytes(4, byteorder="big")
    com_length = len(commitments[0]).to_bytes(4, byteorder="big")

    if hashes is not None:
        hash_length = len(hashes[0])
    else:
        hash_length = 0

    # Prepare hash length and begin packing payload
    hash_length = hash_length.to_bytes(4, byteorder="big")
    message = COMM.encode() + number_of_commitments + hash_length + com_length

    for i in range(len(commitments)):
        message += (
            commitments[i] + hashes[i]
        )  # Works in Shurmann since it has hashes to send
        # message += commitments[i] # Works in mietinnen since it doens't do any hashes

    device.send(message)

def send_nonce_msg(connection: socket.socket, nonce: bytes) -> None:
    """
    Sends a nonce message over a network connection.

    :param connection: The network connection to send the nonce over.
    :param nonce: The nonce to send.
    """
    nonce_size = len(nonce).to_bytes(4, byteorder="big")
    message = NONC.encode() + nonce_size + nonce
    connection.send(message)

def send_pake_msg(connection, msg):
    length_payload = 0
    for m in msg:
        length_payload += len(m) + 4

    payload = length_payload.to_bytes(4, byteorder="big")

    for m in msg:
        payload += len(m).to_bytes(4, byteorder="big")
        payload += m

    outgoing = FPFM.encode() + payload

    connection.send(outgoing)

def send_status(connection: socket.socket, status: bool) -> None:
    """
    Sends a success or failure status over a connection.

    :param connection: The network connection over which to send the status.
    :param status: The boolean status indicating success (True) or failure (False).
    """
    if status:
        msg = SUCC
    else:
        msg = FAIL
    connection.send(msg.encode())

def status_standby(connection: socket.socket, timeout: int) -> Optional[bool]:
    """
    Waits for a success or failure status message within a specified timeout period.

    :param connection: The network connection to receive from.
    :param timeout: The maximum time in seconds to wait for a status.
    :returns: True if success, False if failure, None if timeout.
    """
    status = None
    reference = time.time()

    # Set the socket timeout
    connection.settimeout(timeout)

    try:
        # While process hasn't timed out
        while (time.time() - reference) < timeout:
            # Attempt to receive the command
            command = connection.recv(8)

            if not command:  # Handle empty or closed connection
                continue

            if command == SUCC.encode():  # Check if the command is a success message
                status = True
                break
            elif command == FAIL.encode():  # Check if the command is a failure message
                status = False
                break
    except socket.timeout:
        print("status not received within the timeout period.")
        return None

    return status


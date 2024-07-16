import json
import time
from typing import List, Optional, Tuple, Union
import socket

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
    timestamp = reference

    # While process hasn't timed out
    while (timestamp - reference) < timeout:
        # Check for acknowledgement
        timestamp = time.time()
        command = connection.recv(8)

        if command == None:
            continue
        elif command == SUCC.encode():
            status = True
            break
        elif command == FAIL.encode():
            status = False
            break

    return status


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
    reference = time.time()
    timestamp = reference

    # While process hasn't timed out
    while (timestamp - reference) < timeout:
        # Check for acknowledgement
        timestamp = time.time()
        command = connection.recv(8)

        if command == None:
            continue
        elif command == ACKN.encode():
            acknowledged = True
            break

    return acknowledged


def send_commit(commitments: List[bytes], hashes: List[bytes], device: socket.socket) -> None:
    """
    Sends commitments along with their corresponding hashes over a network connection.

    :param commitments: List of commitments to be sent.
    :param hashes: List of hashes corresponding to the commitments.
    :param device: The network connection to send data over.
    """
    # Prepare number of commitments and their lengths
    number_of_commitments = len(commitments).to_bytes(4, byteorder="big")
    com_length = len(commitments[0]).to_bytes(4, byteorder="big")

    if hashes != None:
        hash_length = len(hashes[0])
    else:
        hash_length = 0

    # Prepare hash length and begin packing payload
    hash_length = hash_length.to_bytes(4, byteorder="big")
    message = COMM.encode() + number_of_commitments + hash_length + com_length

    for i in range(len(commitments)):
        message += hashes[i] + commitments[i]

    device.send(message)


def commit_standby(connection: socket.socket, timeout: int) -> Tuple[Optional[List[bytes]], Optional[List[bytes]]]:
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

    while (timestamp - reference) < timeout:
        timestamp = time.time()
        message = connection.recv(8)

        if message[:8] == COMM.encode():
            # Recieve and unpack the hash
            message = connection.recv(12)

            # Unpack all variable lengths
            number_of_commits = int.from_bytes(message[0:4], byteorder="big")
            hash_length = int.from_bytes(message[4:8], byteorder="big")
            com_length = int.from_bytes(message[8:12], byteorder="big")

            commits = connection.recv(number_of_commits * (hash_length + com_length))
            commitments = []
            hashes = []

            # Extract commitments and hashes, appending to their respective lists
            for i in range(number_of_commits):
                hashes.append(
                    commits[
                        i * (hash_length + com_length) : i * (hash_length + com_length)
                        + hash_length
                    ]
                )
                commitments.append(
                    commits[
                        i * (hash_length + com_length)
                        + hash_length : (i + 1) * (hash_length + com_length)
                    ]
                )
            break

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
    timestamp = reference
    key = None

    # While process hasn't timed out
    while (timestamp - reference) < timeout:
        timestamp = time.time()
        message = connection.recv(12)

        if message == None:
            continue
        else:
            command = message[:8]
            if command == DHKY.encode():
                key_size = int.from_bytes(message[8:], "big")
                key = connection.recv(key_size)
                break

    return key


def send_nonce_msg(connection: socket.socket, nonce: bytes) -> None:
    """
    Sends a nonce message over a network connection.

    :param connection: The network connection to send the nonce over.
    :param nonce: The nonce to send.
    """
    nonce_size = len(nonce).to_bytes(4, byteorder="big")
    message = NONC.encode() + nonce_size + nonce
    connection.send(message)


def get_nonce_msg_standby(connection: socket.socket, timeout: int) -> Optional[bytes]:
    """
    Waits to receive a nonce within a specified timeout period.

    :param connection: The network connection to receive from.
    :param timeout: The maximum time in seconds to wait for the nonce.
    :returns: The received nonce if successful, None otherwise.
    """
    reference = time.time()
    timestamp = reference
    nonce = None

    # While process hasn't timed out
    while (timestamp - reference) < timeout:
        timestamp = time.time()
        message = connection.recv(12)

        if message == None:
            continue
        else:
            command = message[:8]
            if command == NONC.encode():
                nonce_size = int.from_bytes(message[8:], "big")
                nonce = connection.recv(nonce_size)
                break

    return nonce

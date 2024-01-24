import ipaddress
import multiprocessing as mp
import pickle
import select
import socket
import time
import types
from multiprocessing import shared_memory

from zeroconf import ServiceBrowser, ServiceListener, Zeroconf

# Device communcation identifiers
# TODO convert these messages to JSON, having number of iterations, iterations completed
HOST = "host    "
START = "start   "
ACK = "ack     "
COMM = "comm    "


def ack(sock):
    sock.send(ACK.encode())


def wait_for_ack(sock, timeout):
    acked = False

    # Keep track of how long it's taking for ACK
    start_time = time.time()
    current_time = start_time
    # While we haven't timed out for ACK
    while (current_time - start_time) < timeout:
        current_time = time.time()
        # Poll for ACK
        msg = sock.recv(8)
        if msg == None:
            continue
        elif msg.decode() == ACK:
            acked = True
            break

    return acked


def ack_all(participating_sockets):
    # Host acknowledges all clients participating
    for i in range(len(participating_sockets)):
        participating_sockets[i].send(ACK.encode())


def wait_for_all_ack(participating_sockets, timeout):
    acked_sockets = []

    start_time = time.time()
    current_time = start_time
    while (current_time - start_time) < timeout:
        current_time = time.time()
        if len(participating_sockets) == 0:
            break

        outputs = []
        # Divide up participating devices as readable or errors
        readable, writable, exceptional = select.select(
            participating_sockets, outputs, participating_sockets
        )
        for s in readable:
            data = s.recv(8)
            # For readable devices that sent the ACK message
            if data and data.decode() == ACK:
                # Add to the list of acknowledged devices
                acked_sockets.append(s)
                # Remove from participating list as it's acknowledged
                participating_sockets.remove(s)

    return acked_sockets


def send_commitment(commitment, h, participating_sockets):
    # Turn object into byte stream
    pickled_comm = pickle.dumps(commitment)
    length_in_bytes = len(pickled_comm).to_bytes(4, byteorder="big")
    # Send commitment flag, hex, and byte stream to all participants
    msg = COMM.encode() + h + length_in_bytes + pickled_comm
    for i in range(len(participating_sockets)):
        participating_sockets[i].send(msg)


def wait_for_commitment(sock, timeout):
    commitment = None
    h = None

    start_time = time.time()
    current_time = start_time
    while (current_time - start_time) < timeout:
        current_time = time.time()
        msg = sock.recv(76)
        if msg[:8].decode() == COMM:
            h = msg[8:72]  # 64 byte hash
            length_in_bytes = int.from_bytes(msg[72:76], "big")
            comm_msg = sock.recv(length_in_bytes)
            # Unpack the message from byte stream
            commitment = pickle.loads(comm_msg)

    return commitment, h

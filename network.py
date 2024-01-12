import socket
import pickle
import multiprocessing as mp
from multiprocessing import shared_memory
import ipaddress
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf
import types
import select
import time

# Device communcation identifiers
HOST = "host    "  
START = "start   "
ACK = "ack     "
COMM = "comm    "

def ack(sock):
    sock.send(ACK.encode())

def wait_for_ack(sock, timeout):
    acked = False

    start_time = time.time()
    current_time = start_time
    while (current_time - start_time) < timeout:
        current_time = time.time()
        msg = sock.recv(8)
        if msg == None:
            continue
        elif msg.decode() == ACK:
            acked = True
            break

    return acked

def ack_all(participating_sockets):
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
        readable, writable, exceptional = select.select(participating_sockets, outputs, participating_sockets)
        for s in readable:
            data = s.recv(8)
            if data and data.decode() == ACK:
                acked_sockets.append(s)
                participating_sockets.remove(s)

    return acked_sockets

def send_commitment(commitment, h, participating_sockets):
    pickled_comm = pickle.dumps(commitment)
    length_in_bytes = len(pickled_comm).to_byte(4, byteorder='big')
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
            h = msg[8:72] # 64 byte hash
            length_in_bytes = int.from_bytes(msg[72:76], 'big')
            comm_msg = sock.recv(length_in_bytes)
            commitment = pickle.loads(comm_msg)

    return commitment, h

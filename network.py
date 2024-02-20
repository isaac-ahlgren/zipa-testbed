import pickle
import select
import time

# Commands
HOST = "host    "
STRT = "start   "
ACKN = "ack     "
COMM = "comm    "
DHKY = "dhkey   "
NONC = "nonce   "


def ack(connection):
    connection.send(ACKN.encode())



def ack_standby(connection, timeout):
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


#TODO: make it so that the size of the hash can be variable
def send_commit(commitment, hash, device):
    length = len(commitment).to_bytes(4, byteorder='big')
    message = (COMM.encode() + hash + length + commitment)
    device.send(message)

def commit_standby(connection, timeout):
    payload, metadata = None
    reference, timestamp = time.time()

    while (timestamp - reference) < timeout:
        timestamp = time.time()
        # 8 byte command, 64 byte hash, 4 byte length
        message = connection.recv(76)

        if message[:8] == COMM.encode():
            # Unpack and return commitment and its hex
            hash = message[8:72]
            length = int.from_bytes(message[72:76], 'big')
            message = connection.recv(length)
            commitment = message
            break

    return commitment, hash

# UNTESTED CODE
def dh_exchange(connection, key):
    key_size = len(key).to_bytes(4, byteorder='big')
    message = DHKY.encode() + key_size + key
    connection.send(message)


def dh_exchange_standby(connection, timeout):
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
                key_size = int.from_bytes(message[8:], 'big')
                key = connection.recv(key_size)              
                break

    return key

def send_nonce_msg(connection, nonce):
    nonce_size = len(nonce).to_bytes(4, byteorder='big')
    message = NONC.encode() + nonce_size + nonce
    connection.send(message)

def get_nonce_msg_standby(connection, timeout):
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
                nonce_size = int.from_bytes(message[8:], 'big')
                nonce = connection.recv(nonce_size)
                break

    return nonce

import pickle
import select
import time

# Commands
HOST = "host    "
STRT = "start   "
ACKN = "ack     "
COMM = "comm    "
DHKY = "dhkey   "
NONC = "nonce    "


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

def ack_all(participants):
    # Host acknowledges all clients participating
    for participant in participants:
        participant.send(ACKN.encode())

def ack_all_standby(participants, timeout):
    acknowledged = []
    reference = time.time()
    timestamp = reference

    while (timestamp - reference) < timeout:
        # Check for acknowledgement
        timestamp = time.time()
        
        if len(participants) == 0:
            break

        # Tabs on incoming and outgoing connections, and exceptions
        output = []
        readable, writable, exception = select.select(
            participants, output, participants
        )

        for incoming in readable:
            data = incoming.recv(8)
            if data == ACKN.encode():
                acknowledged.append(incoming)
                participants.remove(incoming)
    
    return acknowledged

#TODO: make it so that the size of the hash can be variable
def send_commit(commitment, hash, participants):
    length = len(commitment).to_bytes(4, byteorder='big')
    message = (COMM.encode() + hash + length + commitment)
    for participant in participants:
        participant.send(message)

def commit_standby(connection, timeout):
    commitment = None
    hexadecimal = None
    reference = time.time()
    timestamp = reference

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

def dh_exchange_all(participants, key):
    key_size = len(key).to_bytes(4, byteorder='big')
    message = DHKY.encode() + key_size + key
    for i in range(len(participants)):
        participants[i].send(message)

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

def dh_exchange_standby_all(participants, timeout):
    keys_recieved = dict()
    responded = []
    reference = time.time()
    timestamp = reference

    while (timestamp - reference) < timeout:
        # Check for acknowledgement
        timestamp = time.time()
        
        if len(participants) == 0:
            break

        # Tabs on incoming and outgoing connections, and exceptions
        output = []
        readable, writable, exception = select.select(
            participants, output, participants
        )

        for incoming in readable:
            message = incoming.recv(12)
            command = message[:8]
            if command == DHKY.encode():
                key_size = int.from_bytes(message[8:], 'big')
                key = incoming.recv(key_size)
                ip_addr, port = incoming.getpeername()
                keys_recieved[ip_addr] = key

                responded.append(incoming)
                participants.remove(incoming)
    return responded, keys_recieved

def send_nonce(connection, nonce):
    nonce_size = len(nonce).to_bytes(4, byteorder='big')
    message = NONC.encode() + nonce_size + nonce
    connection.send(message)

def send_nonce_all(participants, nonce):
    nonce_size = len(nonce).to_bytes(4, byteorder='big')
    message = NONC.encode() + nonce_size + nonce
    for i in range(len(participants)):
        participants[i].send(message)

def get_nonce_standby(connection, timeout):
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
            if command == NONC.encode():
                nonce_size = int.from_bytes(message[8:], 'big')
                nonce = connection.recv(nonce_size)
                break

    return nonce

def send_nonce_standby_all(participants, timeout):
    nonces_recieved = dict()
    responded = []
    reference = time.time()
    timestamp = reference

    while (timestamp - reference) < timeout:
        # Check for acknowledgement
        timestamp = time.time()
        
        if len(participants) == 0:
            break

        # Tabs on incoming and outgoing connections, and exceptions
        output = []
        readable, writable, exception = select.select(
            participants, output, participants
        )

        for incoming in readable:
            message = incoming.recv(12)
            command = message[:8]
            if command == NONC.encode():
                nonce_size = int.from_bytes(message[8:], 'big')
                nonce = incoming.recv(nonce_size)
                ip_addr, port = incoming.getpeername()
                nonces_recieved[ip_addr] = nonce

                responded.append(incoming)
                participants.remove(incoming)
    return responded, nonces_recieved
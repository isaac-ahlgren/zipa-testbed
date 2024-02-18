import pickle
import select
import time

# Commands
HOST = "host    "
STRT = "start   "
ACKN = "ack     "
COMM = "comm    "
DHKY = "dhkey   "
HASH = "hash    "


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
        command = connection.recv(8).decode()

        if command == None:
            continue
        elif command == ACKN:
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
            data = incoming.recv(8).decode()
            if data == ACKN:
                acknowledged.append(incoming)
                participants.remove(incoming)
    
    return acknowledged

def send_commit(commitment, hexadecimal, participants):
    # Convert message into bytestream with helpful information and send
    bytestream = pickle.dumps(commitment)
    length = len(bytestream).to_bytes(4, byteorder='big')
    message = (COMM.encode() + hexadecimal + length + bytestream)

    for participant in participants:
        participant.send(message)

def commit_standby(connection, timeout):
    commitment = None
    hexadecimal = None
    reference = time.time()
    timestamp = reference

    while (timestamp - reference) < timeout:
        timestamp = time.time()
        # 8 byte command, 64 byte hex, 4 byte length
        message = connection.recv(76)

        if message[:8].decode() == COMM:
            # Unpack and return commitment and its hex
            hexadecimal = message[8:72]
            length = int.from_bytes(message[72:76], 'big')
            message = connection.recv(length)
            commitment = pickle.loads(message)

    return commitment, hexadecimal

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
            command = message[:8].decode()
            if command == DHKY:
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
            command = message[:8].decode()
            if command == DHKY:
                key_size = int.from_bytes(message[8:], 'big')
                key = incoming.recv(key_size)
                ip_addr, port = incoming.getpeername()
                keys_recieved[ip_addr] = key

                responded.append(incoming)
                participants.remove(incoming)
    return responded, keys_recieved

def send_hash(connection, hash):
    hash_size = len(hash).to_bytes(4, byteorder='big')
    message = HASH.encode() + hash_size + hash
    connection.send(message)

def send_hash_all(participants, hash):
    hash_size = len(hash).to_bytes(4, byteorder='big')
    message = HASH.encode() + hash_size + hash
    for i in range(len(participants)):
        participants[i].send(message)

def get_hash_standby(connection, timeout):
    reference = time.time()
    key = None

    # While process hasn't timed out
    while (timestamp - reference) < timeout:
        timestamp = time.time()
        message = connection.recv(12)

        if message == None:
            continue
        else:
            command = message[:8].decode()
            if command == HASH:
                hash_size = int.from_bytes(message[8:], 'big')
                hash = connection.recv(hash_size)
            break

    return hash

def send_hash_standby_all(participants, timeout):
    hashes_recieved = dict()
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
            if command == HASH:
                hash_size = int.from_bytes(message[8:], 'big')
                hash = incoming.recv(hash_size)
                ip_addr, port = incoming.getpeername()
                hashes_recieved[ip_addr] = hash

                responded.append(incoming)
                participants.remove(incoming)
    return responded, hashes_recieved
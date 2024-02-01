import pickle
import select
import time

# Commands
HOST = "host    "
STRT = "start   "
ACKN = "ack     "
COMM = "comm    "


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

def commit(commitment, hexadecimal, participants):
    # Convert message into bytestream with helpful information and send
    bytestream = pickle.dumps(commitment)
    length = len(bytestream).to_bytes(4, byteorder='big')
    message = (COMM.encode() + hexadecimal + length + bytestream)

    for participant in participants:
        participant.send(message)

def commit_standby(connection, timeout):
    commitment = None
    hash_val = None
    reference = time.time()
    timestamp = reference

    while (timestamp - reference) < timeout:
        timestamp = time.time()
        # 8 byte command, 64 byte hex, 4 byte length
        message = connection.recv(76)

        if message[:8].decode() == COMM:
            # Unpack and return commitment and its hex
            hash_val = message[8:72]
            length = int.from_bytes(message[72:76], 'big')
            message = connection.recv(length)
            commitment = pickle.loads(message)

    return commitment, hash_val
    
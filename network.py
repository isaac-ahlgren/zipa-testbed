import pickle
import select
import time

# Commands
HOST = "host    "
STRT = "start   "
ACKN = "ack     "
COMM = "comm    "
DHKY = "dhkey   "


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


def commit(commitment, commitment_hash, time_start, time_stop, participants):
    # Convert message into bytestream with helpful information and send
    payload = pickle.dumps(commitment)
    metadata = pickle.dumps(
        {"time_start": time_start, "time_stop": time_stop, "hash": commitment_hash}
    )
    payload_length = len(payload).to_bytes(4, byteorder="big")
    metadata_length = len(metadata).to_bytes(4, byteorder="big")
    message = COMM.encode() + payload_length + metadata_length + payload + metadata

    for participant in participants:
        participant.send(message)



def commit_standby(connection, timeout):
    payload, metadata = None
    reference, timestamp = time.time()

    while (timestamp - reference) < timeout:
        timestamp = time.time()
        # 8 byte command, 4 byte payload, 4 metadata
        message = connection.recv(16)

        if message[:8].decode() == COMM:
            # Unpack and return commitment and its metadata
            payload_length = int.from_bytes(message[8:12], byteorder="big")
            metadata_length = int.from_bytes(message[12:16], byteorder="big")
            payload = pickle.loads(connection.recv(payload_length))
            metadata = pickle.loads(connection.recv(metadata_length))

    return payload, metadata


# UNTESTED CODE
def dh_exchange(connection, key):
    key_size = len(key).to_bytes(4, byteorder="big")
    message = DHKY + key_size + key
    connection.send(message)


def dh_exchange_all(participants, key):
    key_size = len(key).to_bytes(4, byteorder="big")
    message = DHKY + key_size + key
    for i in range(len(participants)):
        participants[i].send(message)


def dh_exchange_standby(connection, timeout):
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
            if command == DHKY:
                key_size = int.from_bytes(message[8:], "big")
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
            if command == DHKY:
                key_size = int.from_bytes(message[8:], "big")
                key = incoming.recv(key_size)
                ip_addr, port = incoming.getpeername()
                keys_recieved[ip_addr] = key

                responded.append(incoming)
                participants.remove(incoming)
    return responded, keys_recieved

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
SUCC = "success "
FAIL = "failed  "

def send_status(connection, status):
    if status:
        msg = SUCC
    else:
        msg = FAIL
    connection.send(msg.encode())

def status_standby(connection, timeout):
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

def send_commit(commitments, hashes, device):
    number_of_commitments = len(commitments).to_bytes(4, byteorder='big')
    com_length = len(commitments[0]).to_bytes(4, byteorder='big')

    if hashes != None:
        hash_length = len(hashes[0]).to_bytes(4, byteorder='big')
    else:
        hash_length = 0

    message = COMM.encode() + number_of_commitments + hash_length + com_length
    for i in range(len(commitments)):
        message += hashes[i] + commitments[i]
    device.send(message)

def commit_standby(connection, timeout):
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
            number_of_commits = int.from_bytes(message[0:4], byteorder='big')
            hash_length = int.from_bytes(message[4:8], byteorder='big')
            com_length = int.from_bytes(message[8:12], byteorder='big')

            commits = connection.recv(number_of_commits*(hash_length + com_length))
            commitments = []
            hashes = []
            for i in range(number_of_commits):
                hashes.append(commits[i*(hash_length + com_length):i*(hash_length + com_length) + hash_length])
                commitments.append(commits[i*(hash_length + com_length) + hash_length:(i+1)*(hash_length + com_length)])
            break

    return commitments, hashes

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

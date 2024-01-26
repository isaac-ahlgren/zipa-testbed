import pickle
import select
import time

from zeroconf import ServiceBrowser, ServiceListener, Zeroconf

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
        command = connection.recv(8)

        if command == None:
            continue
        elif command.decode() == ACKN:
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
            if data.decode() == ACKN:
                acknowledged.append(incoming)
                participants.remove(incoming)
    
    return acknowledged

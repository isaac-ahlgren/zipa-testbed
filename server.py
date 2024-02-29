import json
import socket

HOST = "host    "
IP_ADDR = "192.168.1.248"
TARGET_IP_ADDR = ("192.168.1.220", 5005)
SHURMANN_JSON = {
    "protocol": {"name": "shurmann-siggs"},
    "n": 12,
    "k": 8,
    "sensor": "microphone",
    "timeout": 10,
    "time_length": 15,
    "sampling": 44100,
    "maximum": 100,
    "iterations": 0,
}

MIETTINEN_JSON = {
    "protocol": {
        "name": "miettinen",
        "f": 5,
        "w": 5,
        "rel_thresh": 0.1,
        "abs_thresh": 0.5,
        "auth_thresh": 0.9,
        "success_thresh": 10,
        "max_iterations": 50,
    },
    "n": 12,
    "k": 8,
    "sensor": "microphone",
    "timeout": 10,
    "time_length": 15 
}

json_string = json.dumps(MIETTINEN_JSON)

if __name__ == "__main__":
    # Pack up message
    bytestream = json.dumps(MIETTINEN_JSON).encode("utf8")
    length = len(bytestream).to_bytes(4, byteorder="big")
    message = HOST.encode() + length + bytestream
    # print(f"JSON: {JSON}\nLength of JSON's bytestream: {length}\nMessage: {message}")

    # Create socket and connect to client that acts as host
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((IP_ADDR, 5005))
    sock.connect(TARGET_IP_ADDR)

    # Send message
    sock.sendto(message, TARGET_IP_ADDR)

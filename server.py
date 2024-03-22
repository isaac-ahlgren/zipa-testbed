import json
import socket

HOST = "host    "
IP_ADDR = "192.168.1.116"
TARGET_IP_ADDR = ("192.168.1.220", 5005)

SHURMANN = {
    "protocol": {"name": "shurmann-siggs", "window_len": 10000, "band_len": 1000},
    "key_length": 8,
    "parity_symbols": 4,
    "sensor": "microphone",
    "timeout": 10,
}

MIETTINEN = {
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
    "key_length": 8,
    "parity_symbols": 4,
    "sensor": "microphone",
    "timeout": 10,
}

VOLTKEY = {
    "protocol": {
        "name": "voltkey",
        "periods": 16,
        "bins": 8,
    },
    "key_length": 8,
    "parity_symbols": 4,
    "timeout": 10,
    "sensor": "test_sensor" # TODO change to VoltKey when ready
}

if __name__ == "__main__":
    # Pack up message
    bytestream = json.dumps(MIETTINEN).encode("utf8")
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

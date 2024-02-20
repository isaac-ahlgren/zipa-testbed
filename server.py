import socket
import pickle
import socket
import pickle
import json

HOST = "host    "
IP_ADDR = "192.168.1.146"
TARGET_IP_ADDR = ("192.168.1.168", 5005)
JSON = {
    "protocol": {"name": "shurmann-siggs", "n": 16, "k": 4},
    "timeout": 10,
    "duration": 30,
    "sampling": 44100,
    "maximum": 100,
    "iterations": 0,
}

json_string = json.dumps(JSON)

if __name__ == "__main__":
    # Pack up message
    bytestream = json.dumps(JSON).encode('utf8')
    length = len(bytestream).to_bytes(4, byteorder='big')
    message = (HOST.encode() + length + bytestream)
    print(f"JSON: {JSON}\nLength of JSON's bytestream: {length}\nMessage: {message}")

    # Create socket and connect to client that acts as host
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((IP_ADDR, 5005))
    sock.connect(TARGET_IP_ADDR)

    # Send message
    sock.sendto(message, TARGET_IP_ADDR)

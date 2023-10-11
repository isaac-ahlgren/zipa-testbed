import socket
import pickle

class Network:
    def __init__(self, ip, other_ip, is_host):
        self.ip = ip
        self.other_ip = other_ip
        self.is_host = is_host
        self.personal_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self.personal_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.personal_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        if is_host:
            self.personal_sock.bind((ip, 5005))
            self.personal_sock.listen()
            (client_sock, addr) = self.personal_sock.accept()
            self.client_sock = client_sock
            self.client_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            print("Connection Made!")
        else:
            self.personal_sock.connect((other_ip, 5005))
            print("Connected to host")

    def send_start(self):
        print()
        print("Sending Start")
        print()
        self.client_sock.sendto("start".encode(), (self.other_ip, 5005))

    def get_ack(self):
        print()
        print("Polling For ACK")
        print()
        while (1):
            message, address = self.client_sock.recvfrom(8)
            if message is not None:
                break

    def get_bits(self, bit_len):
        print()
        print("Waiting For Bits")
        print()
        while (1):
            message, address = self.client_sock.recvfrom(bit_len)
            if message is not None:
                break

        print()
        print("Bits Recieved")
        print()
        bits = message.decode()
        print(bits)
        return bits

    def send_codeword(self, C):
        print()
        print("Sending Codeword")
        print()
        pickled_C = pickle.dumps(C)
        self.client_sock.sendto(pickled_C, (self.other_ip, 5005))

    def get_auth_token(self, bytes_needed):
        print()
        print("Waiting For Authentication Token")
        print()
        while (1):
            message, address = self.client_sock.recvfrom(bytes_needed)
            if message is not None:
                break
        print()
        print("Token Recieved")
        print()
        token = pickle.loads(message)
        return token

    def get_start(self):
        print()
        print("Polling For Start")
        print()
        while (1):
            message, address = self.personal_sock.recvfrom(8)
            if message is not None:
                break

    def send_ack(self):
        print()
        print("Sending Ack")
        print()
        self.personal_sock.sendto("ack".encode(), (self.other_ip, 5005))

    def send_bits(self, bits):
        print()
        print("Sending Bits")
        print()
        self.personal_sock.sendto(bits.encode(), (self.other_ip, 5005))

    def get_codeword(self, bytes_needed):
        print()
        print("Waiting For Codeword")
        print()
        while (1):
            message, address = self.personal_sock.recvfrom(bytes_needed)
            if message is not None:
                break
        print()
        print("Codeword Recieved")
        print()
        C = pickle.loads(message)
        return C

    def send_auth_token(self, token):
        print()
        print("Sending Authentication Token")
        print()
        pickled_token = pickle.dumps(token)
        self.personal_sock.sendto(pickled_token, (self.other_ip, 5005))

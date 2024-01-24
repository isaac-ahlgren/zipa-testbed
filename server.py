import socket
 
if __name__ == "__main__":
    IP_ADDR = "192.168.1.116"
    TARGET_IP_ADDR = ("192.168.1.172", 5005)
 
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((IP_ADDR, 5005))
 
    sock.connect(TARGET_IP_ADDR)
    sock.sendto("host    shur_n_sigg".encode(), TARGET_IP_ADDR)


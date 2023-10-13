from server import Server
import os

if __name__ == "__main__":
    ip = "192.168.1.3"
    port = 5005
    data_folder = os.getcwd() + "/data"

    s = Server(ip, port, data_folder)
    s.start()

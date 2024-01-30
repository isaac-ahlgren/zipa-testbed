import netifaces as ni

from zipa_sys_ import ZIPA_System

f = open("/home/pi/hostname.txt", "r")
ID = f.read().strip()
f.close()


IP = ni.ifaddresses("eth0")[ni.AF_INET][0]["addr"]  # <--- device's ip address goes here
PORT = 5005
SERVICE_NAME = "_zipa._tcp.local."

UNCOND_HOST = True

if __name__ == "__main__":
    zipa_sys = ZIPA_System(
        ID, IP, PORT, SERVICE_NAME, "/mnt/data")
    zipa_sys.start()

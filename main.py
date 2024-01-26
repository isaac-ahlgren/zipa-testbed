import netifaces as ni

from zipa_sys import ZIPA_System

f = open("/home/pi/hostname.txt", "r")
ID = f.read().strip()
f.close()


# TODO: Make a function that retrieves the JSON and feed it to the zipa system.
IP = ni.ifaddresses("eth0")[ni.AF_INET][0]["addr"]  # <--- device's ip address goes here
PORT = 5005
SAMPLE_RATE = 44100
SECONDS = 30
N = 16
K = 4
SERVICE_NAME = "_zipa._tcp.local."
TIMEOUT = 10

UNCOND_HOST = True

if __name__ == "__main__":
    zipa_sys = ZIPA_System(
        ID, IP, PORT, SERVICE_NAME, "/mnt/data", TIMEOUT, SAMPLE_RATE, SECONDS, N, K
    )
    zipa_sys.start()

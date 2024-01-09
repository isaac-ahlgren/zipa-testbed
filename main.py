from zipa_sys import ZIPA_System

ID = 1
IP = "192.168.1.168"        # <--- device's ip address goes here
SAMPLE_RATE = 44100
SECONDS = 10
N = 16
K = 4
SERVICE_NAME = "_zipa._tcp.local."
TIMEOUT = 10

UNCOND_HOST = True

if __name__ == "__main__":
    zipa_sys = ZIPA_System(ID, IP, SERVICE_NAME, "/mnt/data", TIMEOUT, SAMPLE_RATE, SECONDS, N, K)

    zipa_sys.zipa_protocol(UNCOND_HOST)

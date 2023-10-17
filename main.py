from zipa_sys import ZIPA_System

ID = 0
IP = "192.168.1.8"        # <--- device's ip address goes here
OTHER_IP = "192.168.1.10"  # <--- other device's ip address goes here
SAMPLE_RATE = 44100
SECONDS = 10
N = 16
K = 4
IS_HOST = True
EXP_NAME = "test"

if __name__ == "__main__":
    zipa_sys = ZIPA_System(ID, IS_HOST, IP, OTHER_IP, "/mnt/data", SAMPLE_RATE, SECONDS, "test", N, K)

    if IS_HOST:
        zipa_sys.bit_agreement_exp_host()
    else:
        zipa_sys.bit_agreement_exp_dev()

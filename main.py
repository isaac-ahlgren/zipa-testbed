from tr_sys import TR_Bit_Extract_System

IP = "192.168.1.2"        # <--- device's ip address goes here
OTHER_IP = "192.168.1.4"  # <--- other device's ip address goes here
SAMPLE_RATE = 44100
VECTOR_NUM = 256
EIG_NUM = 4
BINS = 16
SECONDS = 4
N = 16
K = 4
IS_HOST = True
EXP_NAME = "test"

if __name__ == "__main__":
    if IS_HOST:
        tr.bit_agreement_exp_host()
    else:
        tr.bit_agreement_exp_dev()

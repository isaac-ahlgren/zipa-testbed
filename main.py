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
    tr_phone = TR_Bit_Extract_System(IS_HOST, IP, OTHER_IP, SAMPLE_RATE, VECTOR_NUM, EIG_NUM, BINS, SECONDS, EXP_NAME, N, K, 'iPhone')
    tr_laptop = TR_Bit_Extract_System(IS_HOST, IP, OTHER_IP, SAMPLE_RATE, VECTOR_NUM, EIG_NUM, BINS, SECONDS, EXP_NAME, N, K, 'MacBook')
    key_phone, conv_phone = tr_phone.extract_context()
    key_laptop, conv_laptop = tr_laptop.extract_context()
    print(str(key_phone))
    print(str(key_laptop))

    bit_error_count = sum(1 for a, b in zip(str(key_phone), str(key_laptop)) if a != b)
    bit_error_rate = bit_error_count / len(str(key_phone))

    print ("BER = " + str(bit_error_rate))
#    if IS_HOST:
#        tr.bit_agreement_exp_host()
#    else:
#        tr.bit_agreement_exp_dev()

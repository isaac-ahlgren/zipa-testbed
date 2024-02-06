from reed_solomon import *
import random

PRIME_POLY = 0b100011101
GEN_POLY =   0b10000011
BLOCK_SIZE = 8

def error(bits, error_rate):
    b = []
    for i in range(len(bits)):
        if (random.randint(0,99)/100) < error_rate:
            b.append(bits[i] - 1)
        else:
            b.append(bits[i])
    
    b[0] = abs(bits[0] - 1)
    b[1] = abs(bits[1] - 1)
    b[2] = abs(bits[2] - 1)
    b[3] = abs(bits[3] - 1)
    b[4] = abs(bits[4] - 1)

    return bytes(b)

if __name__ == "__main__":
    k = 8
    n = 12
    rs = ReedSolomonObj(n, k, BLOCK_SIZE, PRIME_POLY, GEN_POLY)

    total_successes = 0
    total = 500
    for i in range(total):
        key = random.randbytes(8)
        codeword = rs.encode(key)
        #print(codeword)
        err_codeword = error(codeword, 0)
        #print("codeword " + str(codeword))
        #print("err codeword " + str(err_codeword))
        recieved_key = rs.decode(err_codeword)
        #print(key)
        #print(recieved_key)
        #print()
        if key == recieved_key:
            total_successes += 1
        else:
            print()
            print("end")
            print(key)
            print(recieved_key)
            print()
    print(total_successes/total)


from galois import *
import random

PRIME_POLY = 0b100011101
GEN_POLY =   0b10000011
BLOCK_SIZE = 8

class Reed_Solomon:
    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.block_size = BLOCK_SIZE
        
        self.GF = GaloisField(self.block_size, PRIME_POLY, GEN_POLY)
        self.RS = ReedSolomonObj(self.GF, n, k)
        self.PA = polynomial_arithmetic(self.GF)

    def get_random_coeffs(self, random_bits):
        num_coeffs = int(len(random_bits) / self.block_size)
        random_key_coeff = []
        
        for i in range(0, num_coeffs):
            sub_bits = random_bits[i*self.block_size:(i+1)*self.block_size]
            random_key_coeff.append(int(sub_bits,2))
        return random_key_coeff
    
    def encode_message(self, random_bits):

        random_coeffs = self.get_random_coeffs(random_bits)
    
        g = self.RS.get_generator_poly()

        key = polynomial(self.k)
        key_coeffs = []
        for i in range(0, self.k):
            coeff = random.randint(0, self.GF.max_num - 1)
            key_coeffs.append(coeff)
        key.set_coeffs(key_coeffs)
        key.resize()

        # Multply
        C = self.PA.mult(key, g)

        for i in range(len(C.coeffs)):
            C.coeffs[i] ^= random_coeffs[i]
        return key, C

    def decode_message(self, C, random_bits):
        random_coeffs = self.get_random_coeffs(random_bits)
    
        g = self.RS.get_generator_poly()

        for i in range(len(C.coeffs)):
            C.coeffs[i] ^= random_coeffs[i]

        # Collect syndrome polynomial
        poly,syndromes = self.RS.calculate_syndrome(C, g)

        if type(poly) != type(0):
            sig = self.RS.berlecamp_alg(poly, syndromes)
        else:
            sig = 0
            return C

        s_r = self.RS.get_sigma_r(sig)
        zeros = s_r.find_zeros(self.GF)
        found_errors = self.RS.find_error_values(poly,zeros)
        output = self.RS.correct_found_errors(C,zeros,found_errors)
        output.resize()

        return output

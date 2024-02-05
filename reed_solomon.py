import random
from galois import *

#TODO: Something is wrong with this implementation I think, with parameters n = 12 and k = 8, it can correct 4 symbol errors when it only should be 2
# See test.py to see what I mean

# Currently built to assume block size will be 8 bits
PRIME_POLY = 0b100011101
GEN_POLY =   0b10000011
BLOCK_SIZE = 8
class ReedSolomonObj():
    def __init__(self, n, k, block_size, prime_poly, gen_poly):
        if k > n:
            raise Exception("k has to be this relation to n, k <= n")

        self.n = n
        self.k = k
        self.t = int((n-k)/2)
        self.block_size = BLOCK_SIZE
        self.prime_poly = PRIME_POLY
        self.gen_poly = GEN_POLY
        self.field = GaloisField(self.block_size, self.prime_poly, self.gen_poly)
        self.PA = polynomial_arithmetic(self.field)

    def get_bytes_to_poly(self, b: bytes) -> polynomial:
        coeffs = []
        
        for i in range(len(b)):
            coeffs.append(b[i])
        data = polynomial(len(b))
        data.set_coeffs(coeffs)
        data.resize()
        return data

    def encode(self, key: bytes) -> bytes:
        key_poly = self.get_bytes_to_poly(key)

        g = self.get_generator_poly()

        # Multply
        C = self.PA.mult(key_poly, g)
 
        return C.get_bytes(self.n)

    def decode(self, C: bytes) -> bytes:
        C_poly = self.get_bytes_to_poly(C)
        g = self.get_generator_poly()

        # Collect syndrome polynomial
        poly, syndromes = self.calculate_syndrome(C_poly, g)
        if type(poly) != type(0):
            sig = self.berlecamp_alg(poly, syndromes)
        else:
            data = self.PA.div(C_poly, g)
            data.resize()
            return data.get_bytes(self.k)

        s_r = self.get_sigma_r(sig)
        zeros = s_r.find_zeros(self.field)
        found_errors = self.find_error_values(poly,zeros)
        output = self.correct_found_errors(C_poly,zeros,found_errors)
        output.resize()

        data = self.PA.div(output, g)
        data.resize()
        return data.get_bytes(self.k)

    def calculate_syndrome(self, C, g):
        t = self.t
        synds = 0
        S = polynomial(2*t)
        S_coeffs = []

        for i in range(0, 2*t):
            S_coeffs.append(0)

        if type(g) != type(polynomial(self.field)) or type(C) != type(polynomial(self.field)):
            raise Exception('C and g both must be polynomial objects.')
        s = self.PA.div(C, g, 1)

        if s.is_zero():
            return 0,0
        iter = 1    

        for i in range(1, 2*t+1):
            S_coeffs[i-1] = self.field.eval_poly(s, self.field.pow(self.field.generator, i))
            for j in range(0, i):
                if(S_coeffs[i-1] == S_coeffs[j] and i-1 != j):
                    #print(f"{S_coeffs} {i-1} {j}")
                    iter = 0
            if iter == 1:
                synds+=1
            iter = 1
        S.set_coeffs(S_coeffs)
        return (S,synds)

    def berlecamp_alg(self, S, t):
        C = polynomial(2)
        B = polynomial(2)
        co1 = [1,0]
        co2 = [1,0]

        C.set_coeffs(co1)
        B.set_coeffs(co2)

        L = 0
        m = 1
        b = 1


        if(t <= len(S.coeffs)-1):
            t= len(S.coeffs)

        for n in range(0, t):
            d = S.coeffs[n]
            for i in range(1, L+1):
                d ^= self.field.mult(C.coeffs[i], S.coeffs[n-i])

            if d == 0:
                m = m+1
            elif 2*L <= n+1:
                T = C
                coeff = self.field.mult(d, self.field.get_inverse(b))
                tmp = polynomial(m+1)
                
                tmp_coeffs = []

                #print(f"d = {d} b = {b} coeff = {coeff} m = {m} C = {C} B = {B}")
                for i in range(0, m+1):
                    tmp_coeffs.append(0)
                tmp_coeffs[m] = coeff
                tmp.set_coeffs(tmp_coeffs)

                C = self.PA.add(C, self.PA.mult(tmp, B))
                L = n+1 - L
                B = T
                b =d
                m = 1
                #print(f"d = {d} b = {b} coeff = {coeff} m = {m} C = {C} B = {B} tmp = {tmp}")
            else:
                coeff = self.field.mult(d, self.field.get_inverse(b))
                tmp = polynomial(m+1)
                tmp_coeffs = []
                for i in range(0, m+1):
                    tmp_coeffs.append(0)
                tmp_coeffs[m] = coeff
                tmp.set_coeffs(tmp_coeffs)
                C= self.PA.add(C, self.PA.mult(tmp, B))
                m += 1

        if L == 0:
            return 0
        
        return C

  
    def get_sigma_r(self, s):
        pos = 0
        size = s.size
        sig_r = polynomial(size)
        sig_r_coeffs = []
        
        for i in range(0, size):
            sig_r_coeffs.append(0)

        for i in range(size-1, -1, -1):
            sig_r_coeffs[pos] = s.coeffs[i]
            pos+=1
        sig_r.set_coeffs(sig_r_coeffs)
        
        return sig_r

    def get_generator_poly(self):
        t = self.t
        PA = polynomial_arithmetic(self.field)
        tmp = polynomial(2)
        coeffs = [self.field.generator,1]
        tmp.set_coeffs(coeffs)

        for i in range(2, (2*t)+1):
            tmp_2 = polynomial(2)
            coeffs = [self.field.pow(self.field.generator, i), 1]
            tmp_2.set_coeffs(coeffs)
            tmp = PA.mult(tmp, tmp_2)
        
        # tmp.coeffs.reverse()
        g = tmp

        return g 

    def find_error_values(self, C, roots):
        matrix = []
        count = 0
        Gauss = GaussianObj(self.field)

        B = []

        for i in range(0, len(roots)):
            A = []
            for j in range(0,len(roots)+1):
                if j < len(roots):
                    A.append(self.field.pow(self.field.pow(self.field.generator, roots[j]), count+1))
                else:
                    A.append(C.coeffs[i])
                    
            count+=1
            matrix.append(A)
        sols = Gauss.solve_system(matrix)
        return sols

    def correct_found_errors(self,C,locations,errors):
        count = 0
        for i in locations:
            #print(f"count = {count} i = {i}")
            if i < len(C.coeffs):
                C.coeffs[i] ^= errors[count]
            count+=1
        C.resize()
        return C

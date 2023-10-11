# Example Prime poly and generator
# unsigned int prim_poly = (unsigned int) 0b100011101;
# int generator = (unsigned int)0b10000011;
import random
import numpy as np
from gauss import *

class polynomial():
    def __init__(self, size, poly=None):
        if poly:
            self.coeffs = []
            self.size = poly.size
            self.set_coeffs(poly.coeffs)
        else:
            self.coeffs = []
            self.size = size
        

    def set_coeffs(self, coeffs):
        for num in coeffs:
            if (len(self.coeffs) < self.size):
                self.coeffs.append(num)
        
    
    def set_size(self, size):
        self.size = size

    def find_zeros(self, field):
        max_num = field.max_num
        zeros = []

        for i in range(0, max_num):
            num = field.eval_poly(self, field.pow(field.generator, i))
            if num == 0:
                zeros.append(i) 
        
        return zeros


    def resize(self):
        new_coeffs = []
        check = True
        start = 0
        for i in range(self.size-1, -1, -1):
            if check:
                if self.coeffs[i] != 0:
                    start = i
                    check = False
                else:
                    continue
        for i in range(0, start+1):
            new_coeffs.append(self.coeffs[i])
        
        self.coeffs = new_coeffs
        self.size = len(self.coeffs)

    def __str__(self):
        string = ''
        count = 0
        for i in self.coeffs:
            string += f" {i}x^{count} + "
            count+=1
        return string


class polynomial_arithmetic():
    def __init__(self, field):
        self.field = field

    def add(self, f, g):
        if f.size > g.size:
            poly_len = f.size
        else:
            poly_len = g.size
        
        new_poly = polynomial(poly_len)
        new_coeffs = []
        for i in range(0, poly_len):
            new_coeffs.append(0)

        for i in range(poly_len-1, -1, -1):
            if f.size > i:
                new_coeffs[i] ^= f.coeffs[i]
                
            if g.size > i:
                new_coeffs[i] ^= g.coeffs[i]
              
            
    
        new_poly.set_coeffs(new_coeffs)
        return new_poly

    
    def mult(self, f, g):
        if type(f) == type(polynomial(0)) and type(f) == type(g):
            new_poly = polynomial(f.size + g.size - 1)
            new_coeffs = []
            for i in range(0,(f.size + g.size)):
                new_coeffs.append(0)
            i = f.size-1
            for co1 in f.coeffs:
                j = g.size-1
                for co2 in g.coeffs:
                    new_coeffs[i+j] ^= self.field.mult(g.coeffs[j],f.coeffs[i])
                    #print(f"Multiplying {g.coeffs[j]}*{f.coeffs[i]}x^{i+j} = {new_coeffs[i+j]}")
                    j-=1
                i -= 1
            new_poly.set_coeffs(new_coeffs)
            new_poly.resize()
            return new_poly

    #we divide f/g        
    #the remainder argument tells the function you would like to return the remainder rather than the quotient
    def div(self, f, g, remainder=0):
        # if there is a dvide by zero error or one of the polynomials are zero
        zero_poly = polynomial(1)
        zeros = [0]
        zero_poly.set_coeffs(zeros)

        if f.coeffs == g.coeffs:
            single_poly = polynomial(1)
            single_poly.set_coeffs([1])
            return single_poly

        quotient = polynomial(f.size)
        new_coeffs = []
        for i in range(0, f.size):
            new_coeffs.append(0)
        quotient.set_coeffs(new_coeffs)
        dividend = polynomial(f.size, poly=f)
        divisor = polynomial(g.size, poly=g)

        for i in range(0, f.size):
            new_coeffs.append(0)

        if type(f) == type(polynomial(0)) and type(f) == type(g):
            numerator_size = f.size
            denomenator_size = g.size
            multiplier = 0
            coeff = 0
            dividend_position = f.size-1
            quotient_pos = 0
            while (dividend_position >= g.size-1):
                current_coeff = dividend.coeffs[dividend_position]
                #finding the constant to multiply the function by
                for i in range(0, self.field.max_num):
                    const = self.field.mult(i, g.coeffs[-1])
                    if (const ^ current_coeff == 0):
                        coeff = i
                        const = i
                        break
                
                quotient.coeffs[dividend_position-(g.size-1)] = coeff
                if coeff != 0:
                    tmp_pos = 0
                    for j in range(g.size-1, -1,-1):
                        if(g.coeffs[j] != 0 ):
                            dividend.coeffs[dividend_position-tmp_pos] ^= self.field.mult(g.coeffs[j], const)
                            # dividend.resize()
                        tmp_pos+=1

                dividend_position -= 1
        else:
            return zero_poly

        quotient.resize()
        dividend.resize()

        if remainder == 0:
            return quotient
        else:
            return dividend


class GaloisField():
    def __init__(self, size, prime_poly, generator):
        self.size = size
        self.max_num = 2**size
        self.prime_poly = prime_poly
        self.generator = generator
        self.gflog = [0 for i in range(0, 2**size)]
        self.gfilog = [ 0 for i in range(0, 2**size)]

        self.setup_tables()
    
    def setup_tables(self):
        b = 1
        log = 0
        for i in range(0, 2**self.size):
            log = i
            self.gflog[b] = log
            self.gfilog[log] = b
            b = b << 1
            if(b & self.max_num == 2**self.size):
                b = (b ^ self.prime_poly)

    def add(self, x, y):
        if x > self.max_num or y > self.max_num:
            return 0
        return (x^y)

    def mult(self, x, y):
        if x == 0 or y == 0:
            return 0
        if (x >= self.max_num or y >= self.max_num):
            return 0
        sum_log = self.gflog[x] + self.gflog[y]
        #print(f"The product is {self.gfilog[sum_log]}")
        if (sum_log >= self.max_num):
            sum_log -= self.max_num-1;
            if sum_log >= self.max_num:
                return 0
        
        #print(f"Sum_log: {sum_log} = {self.gflog[x]} + {self.gflog[y]}")
        #print(f"The product of {x} * {y} =  {self.gfilog[sum_log]}")
        return self.gfilog[sum_log]

    def div(self, x, y):
        if(x == 0 or y == 0):
            return 0
        diff_log = self.gflog[x] - self.gflog[y]
        if(diff_log < 0):
            diff_log += self.max_num-1
        return self.gfilog[diff_log]

    #x^a
    def pow(self, x, a):
        if x == 0:
            return 0
        res = 1
        for i in range(1, a+1):
            res = self.mult(res, x)
        return res

    def eval_poly(self, p, val):
        if type(p) != type(polynomial(self)):
            if type(p) == type(1):
                return p
            else:
                raise Exception('p argument has to be a polynomial object.')
        
        res = p.coeffs[0]
        pos = 0
        for coeff in p.coeffs:
            if pos > 0:
                res ^= self.mult(coeff, self.pow(val,pos))
            pos+=1
        
        return res

    def get_inverse(self, x):
        
        for i in range(1, self.max_num):
            tmp = self.mult(x,i)
            if tmp == 1:
                return i
        
        return -1

    def __str__(self):
        return f"Gflog = {self.gflog}\t Gfilog = {self.gfilog}"


class ReedSolomonObj():
    def __init__(self, field, n, k):
        if k > n:
            raise Exception("k has to be this relation to n, k <= n")

        self.n = n
        self.k = k
        self.t = int((n-k)/2)
        self.field = field
        self.PA = polynomial_arithmetic(field)

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
        if s.coeffs == [1]:
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
        
        C.resize()
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
        sig_r.resize()
        
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
        Guass = GaussianObj(self.field)

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
        sols = Guass.solve_system(matrix)
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


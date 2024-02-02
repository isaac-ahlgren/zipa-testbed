
#TODO: Create function for creating codeword for key and decoding codeword for key
class ReedSolomonObj():
    def __init__(self, n, k, block_size, prime_poly, gen_poly):
        if k > n:
            raise Exception("k has to be this relation to n, k <= n")

        self.n = n
        self.k = k
        self.t = int((n-k)/2)
        self.block_size = block_size
        self.prime_poly = prime_poly
        self.gen_poly = gen_poly
        self.field = GaloisField(self.block_size, self.prime_poly, self.gen_poly)
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

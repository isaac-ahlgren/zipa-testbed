import os
import random
from copy import copy

import Crypto.Util.number as nb

"""
Thanks to https://mortendahl.github.io/2017/08/13/secret-sharing-part3/

"""


def list_to_byte(list):
    acc = b""
    for b in list:
        acc += b
    return acc


def bytes_to_int(bytes):
    return int.from_bytes(bytes, "little")


def int_to_bytes(integer, n):
    return integer.to_bytes(n, "little")


class SimpleReedSolomonObj:
    """
    Robust Shamir Secret Sharing implementation from https://mortendahl.github.io/2017/08/13/secret-sharing-part3/
    Isaac: Also helpful - https://research.swtch.com/field
    https://en.wikiversity.org/wiki/Reed%E2%80%93Solomon_codes_for_coders/Additional_information#Universal_Reed-Solomon_Codec
    """

    def __init__(self, n, k, power_of_2, prime_poly):
        self.field_size = 2**power_of_2
        if k > n:
            raise Exception("k has to be this relation to n, k <= n")
        if n > self.field_size:
            raise Exception(
                "n has to be n <= 2^(power_of_2) to be valid"
            )
        self.block_byte_size = self.field_size // 8
        self.prime_poly = prime_poly
        self.n = n
        self.k = k
        self.t = n - k

    def rs_generator_poly(nsym, fcr=0, generator=2):
        '''Generate an irreducible generator polynomial (necessary to encode a message into Reed-Solomon)'''
        g = [1]
        for i in xrange(nsym):
            g = gf_poly_mul(g, [1, gf_pow(generator, i+fcr)])
        return g

    def rs_generator_poly_all(max_nsym, fcr=0, generator=2):
        '''Generate all irreducible generator polynomials up to max_nsym (usually you can use n, the length of the message+ecc). Very useful to reduce processing time if you want to encode using variable schemes and nsym rates.'''
        g_all = {}
        g_all[0] = g_all[1] = [1]
        for nsym in xrange(max_nsym):
            g_all[nsym] = rs_generator_poly(nsym, fcr, generator)
        return g_all

    def rs_simple_encode_msg(msg_in, nsym, fcr=0, generator=2, gen=None):
        '''Simple Reed-Solomon encoding (mainly an example for you to understand how it works, because it's slower than the inlined function below)'''
        global field_charac
        if (len(msg_in) + nsym) > field_charac: raise ValueError("Message is too long (%i when max is %i)" % (len(msg_in)+nsym, field_charac))
        if gen is None: gen = rs_generator_poly(nsym, fcr, generator)

        # Pad the message, then divide it by the irreducible generator polynomial
        _, remainder = gf_poly_div(msg_in + [0] * (len(gen)-1), gen)
        # The remainder is our RS code! Just append it to our original message to get our full codeword (this represents a polynomial of max 256 terms)
        msg_out = msg_in + remainder
        # Return the codeword
        return msg_out

    def rs_encode_msg(msg_in, nsym, fcr=0, generator=2, gen=None):
        '''Reed-Solomon main encoding function, using polynomial division (Extended Synthetic Division, the fastest algorithm available to my knowledge), better explained at http://research.swtch.com/field'''
        global field_charac
        if (len(msg_in) + nsym) > field_charac: raise ValueError("Message is too long (%i when max is %i)" % (len(msg_in)+nsym, field_charac))
        if gen is None: gen = rs_generator_poly(nsym, fcr, generator)
        # Init msg_out with the values inside msg_in and pad with len(gen)-1 bytes (which is the number of ecc symbols).
        msg_out = [0] * (len(msg_in) + len(gen)-1)
        # Initializing the Synthetic Division with the dividend (= input message polynomial)
        msg_out[:len(msg_in)] = msg_in

        # Synthetic division main loop
        for i in xrange(len(msg_in)):
            # Note that it's msg_out here, not msg_in. Thus, we reuse the updated value at each iteration
            # (this is how Synthetic Division works: instead of storing in a temporary register the intermediate values,
            # we directly commit them to the output).
            coef = msg_out[i]

            # log(0) is undefined, so we need to manually check for this case.
            if coef != 0:
                # in synthetic division, we always skip the first coefficient of the divisior, because it's only used to normalize the dividend coefficient (which is here useless since the divisor, the generator polynomial, is always monic)
                for j in xrange(1, len(gen)):
                    #if gen[j] != 0: # log(0) is undefined so we need to check that, but it slow things down in fact and it's useless in our case (reed-solomon encoding) since we know that all coefficients in the generator are not 0
                    msg_out[i+j] ^= gf_mul(gen[j], coef) # equivalent to msg_out[i+j] += gf_mul(gen[j], coef)

        # At this point, the Extended Synthetic Divison is done, msg_out contains the quotient in msg_out[:len(msg_in)]
        # and the remainder in msg_out[len(msg_in):]. Here for RS encoding, we don't need the quotient but only the remainder
        # (which represents the RS code), so we can just overwrite the quotient with the input message, so that we get
        # our complete codeword composed of the message + code.
        msg_out[:len(msg_in)] = msg_in

        return msg_out

    def rs_correct_msg_nofsynd(msg_in, nsym, fcr=0, generator=2, erase_pos=None, only_erasures=False):
        '''Reed-Solomon main decoding function, without using the modified Forney syndromes
        This demonstrates how the decoding process is done without using the Forney syndromes
        (this is the most common way nowadays, avoiding Forney syndromes require to use a modified Berlekamp-Massey
        that will take care of the erasures by itself, it's a simple matter of modifying some initialization variables and the loop ranges)'''
        global field_charac
        if len(msg_in) > field_charac:
            raise ValueError("Message is too long (%i when max is %i)" % (len(msg_in), field_charac))

        msg_out = list(msg_in)     # copy of message
        # erasures: set them to null bytes for easier decoding (but this is not necessary, they will be corrected anyway, but debugging will be easier with null bytes because the error locator polynomial values will only depend on the errors locations, not their values)
        if erase_pos is None:
            erase_pos = []
        else:
            for e_pos in erase_pos:
                msg_out[e_pos] = 0
        # check if there are too many erasures
        if len(erase_pos) > nsym: raise ReedSolomonError("Too many erasures to correct")
        # prepare the syndrome polynomial using only errors (ie: errors = characters that were either replaced by null byte or changed to another character, but we don't know their positions)
        synd = rs_calc_syndromes(msg_out, nsym, fcr, generator)
        # check if there's any error/erasure in the input codeword. If not (all syndromes coefficients are 0), then just return the codeword as-is.
        if max(synd) == 0:
            return msg_out[:-nsym], msg_out[-nsym:]  # no errors

        # prepare erasures locator and evaluator polynomials
        erase_loc = None
        #erase_eval = None
        erase_count = 0
        if erase_pos:
            erase_count = len(erase_pos)
            erase_pos_reversed = [len(msg_out)-1-eras for eras in erase_pos]
            erase_loc = rs_find_errata_locator(erase_pos_reversed, generator=generator)
            #erase_eval = rs_find_error_evaluator(synd[::-1], erase_loc, len(erase_loc)-1)

        # prepare errors/errata locator polynomial
        if only_erasures:
            err_loc = erase_loc[::-1]
            #err_eval = erase_eval[::-1]
        else:
            err_loc = rs_find_error_locator(synd, nsym, erase_loc=erase_loc, erase_count=erase_count)
            err_loc = err_loc[::-1]
            #err_eval = rs_find_error_evaluator(synd[::-1], err_loc[::-1], len(err_loc)-1)[::-1] # find error/errata evaluator polynomial (not really necessary since we already compute it at the same time as the error locator poly in BM)

        # locate the message errors
        err_pos = rs_find_errors(err_loc, len(msg_out), generator) # find the roots of the errata locator polynomial (ie: the positions of the errors/errata)
        if err_pos is None:
            raise ReedSolomonError("Could not locate error")

        # compute errata evaluator and errata magnitude polynomials, then correct errors and erasures
        msg_out = rs_correct_errata(msg_out, synd, err_pos, fcr=fcr, generator=generator)
        # check if the final message is fully repaired
        synd = rs_calc_syndromes(msg_out, nsym, fcr, generator)
        if max(synd) > 0:
            raise ReedSolomonError("Could not correct message")
        # return the successfully decoded message
        return msg_out[:-nsym], msg_out[-nsym:] # also return the corrected ecc block so that the user can check()

    def rs_check(msg, nsym, fcr=0, generator=2):
        '''Returns true if the message + ecc has no error of false otherwise (may not always catch a wrong decoding or a wrong message, particularly if there are too many errors -- above the Singleton bound --, but it usually does)'''
        return ( max(rs_calc_syndromes(msg, nsym, fcr, generator)) == 0 )

    def rs_calc_syndromes(msg, nsym, fcr=0, generator=2):
        '''Given the received codeword msg and the number of error correcting symbols (nsym), computes the syndromes polynomial.
        Mathematically, it's essentially equivalent to a Fourrier Transform (Chien search being the inverse).
        '''
        # Note the "[0] +" : we add a 0 coefficient for the lowest degree (the constant). This effectively shifts the syndrome, and will shift every computations depending on the syndromes (such as the errors locator polynomial, errors evaluator polynomial, etc. but not the errors positions).
        # This is not necessary, you can adapt subsequent computations to start from 0 instead of skipping the first iteration (ie, the often seen range(1, n-k+1)),
        synd = [0] * nsym
        for i in xrange(nsym):
            synd[i] = gf_poly_eval(msg, gf_pow(generator, i+fcr))
        return [0] + synd # pad with one 0 for mathematical precision (else we can end up with weird calculations sometimes)

    def rs_correct_errata(msg_in, synd, err_pos, fcr=0, generator=2): # err_pos is a list of the positions of the errors/erasures/errata
        '''Forney algorithm, computes the values (error magnitude) to correct the input message.'''
        global field_charac
        # calculate errata locator polynomial to correct both errors and erasures (by combining the errors positions given by the error locator polynomial found by BM with the erasures positions given by caller)
        coef_pos = [len(msg_in) - 1 - p for p in err_pos] # need to convert the positions to coefficients degrees for the errata locator algo to work (eg: instead of [0, 1, 2] it will become [len(msg)-1, len(msg)-2, len(msg) -3])
        err_loc = rs_find_errata_locator(coef_pos, generator)
        # calculate errata evaluator polynomial (often called Omega or Gamma in academic papers)
        err_eval = rs_find_error_evaluator(synd[::-1], err_loc, len(err_loc)-1)[::-1]

        # Second part of Chien search to get the error location polynomial X from the error positions in err_pos (the roots of the error locator polynomial, ie, where it evaluates to 0)
        X = [] # will store the position of the errors
        for i in xrange(len(coef_pos)):
            l = field_charac - coef_pos[i]
            X.append( gf_pow(generator, -l) )

        # Forney algorithm: compute the magnitudes
        E = [0] * (len(msg_in)) # will store the values that need to be corrected (substracted) to the message containing errors. This is sometimes called the error magnitude polynomial.
        Xlength = len(X)
        for i, Xi in enumerate(X):

            Xi_inv = gf_inverse(Xi)

            # Compute the formal derivative of the error locator polynomial (see Blahut, Algebraic codes for data transmission, pp 196-197).
            # the formal derivative of the errata locator is used as the denominator of the Forney Algorithm, which simply says that the ith error value is given by error_evaluator(gf_inverse(Xi)) / error_locator_derivative(gf_inverse(Xi)). See Blahut, Algebraic codes for data transmission, pp 196-197.
            err_loc_prime_tmp = []
            for j in xrange(Xlength):
                if j != i:
                    err_loc_prime_tmp.append( gf_sub(1, gf_mul(Xi_inv, X[j])) )
            # compute the product, which is the denominator of the Forney algorithm (errata locator derivative)
            err_loc_prime = 1
            for coef in err_loc_prime_tmp:
                err_loc_prime = gf_mul(err_loc_prime, coef)
            # equivalent to: err_loc_prime = functools.reduce(gf_mul, err_loc_prime_tmp, 1)

            # Compute y (evaluation of the errata evaluator polynomial)
            # This is a more faithful translation of the theoretical equation contrary to the old forney method. Here it is an exact reproduction:
            # Yl = omega(Xl.inverse()) / prod(1 - Xj*Xl.inverse()) for j in len(X)
            y = gf_poly_eval(err_eval[::-1], Xi_inv) # numerator of the Forney algorithm (errata evaluator evaluated)
            y = gf_mul(gf_pow(Xi, 1-fcr), y) # adjust to fcr parameter
            
            # Check: err_loc_prime (the divisor) should not be zero.
            if err_loc_prime == 0:
                raise ReedSolomonError("Could not find error magnitude")    # Could not find error magnitude

            # Compute the magnitude
            magnitude = gf_div(y, err_loc_prime) # magnitude value of the error, calculated by the Forney algorithm (an equation in fact): dividing the errata evaluator with the errata locator derivative gives us the errata magnitude (ie, value to repair) the ith symbol
            E[err_pos[i]] = magnitude # store the magnitude for this error into the magnitude polynomial

        # Apply the correction of values to get our message corrected! (note that the ecc bytes also gets corrected!)
        # (this isn't the Forney algorithm, we just apply the result of decoding here)
        msg_in = gf_poly_add(msg_in, E) # equivalent to Ci = Ri - Ei where Ci is the correct message, Ri the received (senseword) message, and Ei the errata magnitudes (minus is replaced by XOR since it's equivalent in GF(2^p)). So in fact here we substract from the received message the errors magnitude, which logically corrects the value to what it should be.
        return msg_in

    def rs_find_error_locator(synd, nsym, erase_loc=None, erase_count=0):
        '''Find error/errata locator and evaluator polynomials with Berlekamp-Massey algorithm'''
        # The idea is that BM will iteratively estimate the error locator polynomial.
        # To do this, it will compute a Discrepancy term called Delta, which will tell us if the error locator polynomial needs an update or not
        # (hence why it's called discrepancy: it tells us when we are getting off board from the correct value).

        # Init the polynomials
        if erase_loc: # if the erasure locator polynomial is supplied, we init with its value, so that we include erasures in the final locator polynomial
            err_loc = list(erase_loc)
            old_loc = list(erase_loc)
        else:
            err_loc = [1] # This is the main variable we want to fill, also called Sigma in other notations or more formally the errors/errata locator polynomial.
            old_loc = [1] # BM is an iterative algorithm, and we need the errata locator polynomial of the previous iteration in order to update other necessary variables.
        #L = 0 # update flag variable, not needed here because we use an alternative equivalent way of checking if update is needed (but using the flag could potentially be faster depending on if using length(list) is taking linear time in your language, here in Python it's constant so it's as fast.

        # Fix the syndrome shifting: when computing the syndrome, some implementations may prepend a 0 coefficient for the lowest degree term (the constant). This is a case of syndrome shifting, thus the syndrome will be bigger than the number of ecc symbols (I don't know what purpose serves this shifting). If that's the case, then we need to account for the syndrome shifting when we use the syndrome such as inside BM, by skipping those prepended coefficients.
        # Another way to detect the shifting is to detect the 0 coefficients: by definition, a syndrome does not contain any 0 coefficient (except if there are no errors/erasures, in this case they are all 0). This however doesn't work with the modified Forney syndrome, which set to 0 the coefficients corresponding to erasures, leaving only the coefficients corresponding to errors.
        synd_shift = 0
        if len(synd) > nsym: synd_shift = len(synd) - nsym

        for i in xrange(nsym-erase_count): # generally: nsym-erase_count == len(synd), except when you input a partial erase_loc and using the full syndrome instead of the Forney syndrome, in which case nsym-erase_count is more correct (len(synd) will fail badly with IndexError).
            if erase_loc: # if an erasures locator polynomial was provided to init the errors locator polynomial, then we must skip the FIRST erase_count iterations (not the last iterations, this is very important!)
                K = erase_count+i+synd_shift
            else: # if erasures locator is not provided, then either there's no erasures to account or we use the Forney syndromes, so we don't need to use erase_count nor erase_loc (the erasures have been trimmed out of the Forney syndromes).
                K = i+synd_shift

            # Compute the discrepancy Delta
            # Here is the close-to-the-books operation to compute the discrepancy Delta: it's a simple polynomial multiplication of error locator with the syndromes, and then we get the Kth element.
            #delta = gf_poly_mul(err_loc[::-1], synd)[K] # theoretically it should be gf_poly_add(synd[::-1], [1])[::-1] instead of just synd, but it seems it's not absolutely necessary to correctly decode.
            # But this can be optimized: since we only need the Kth element, we don't need to compute the polynomial multiplication for any other element but the Kth. Thus to optimize, we compute the polymul only at the item we need, skipping the rest (avoiding a nested loop, thus we are linear time instead of quadratic).
            # This optimization is actually described in several figures of the book "Algebraic codes for data transmission", Blahut, Richard E., 2003, Cambridge university press.
            delta = synd[K]
            for j in xrange(1, len(err_loc)):
                delta ^= gf_mul(err_loc[-(j+1)], synd[K - j]) # delta is also called discrepancy. Here we do a partial polynomial multiplication (ie, we compute the polynomial multiplication only for the term of degree K). Should be equivalent to brownanrs.polynomial.mul_at().
            #print "delta", K, delta, list(gf_poly_mul(err_loc[::-1], synd)) # debugline

            # Shift polynomials to compute the next degree
            old_loc = old_loc + [0]

            # Iteratively estimate the errata locator and evaluator polynomials
            if delta != 0: # Update only if there's a discrepancy
                if len(old_loc) > len(err_loc): # Rule B (rule A is implicitly defined because rule A just says that we skip any modification for this iteration)
                #if 2*L <= K+erase_count: # equivalent to len(old_loc) > len(err_loc), as long as L is correctly computed
                    # Computing errata locator polynomial Sigma
                    new_loc = gf_poly_scale(old_loc, delta)
                    old_loc = gf_poly_scale(err_loc, gf_inverse(delta)) # effectively we are doing err_loc * 1/delta = err_loc // delta
                    err_loc = new_loc
                    # Update the update flag
                    #L = K - L # the update flag L is tricky: in Blahut's schema, it's mandatory to use `L = K - L - erase_count` (and indeed in a previous draft of this function, if you forgot to do `- erase_count` it would lead to correcting only 2*(errors+erasures) <= (n-k) instead of 2*errors+erasures <= (n-k)), but in this latest draft, this will lead to a wrong decoding in some cases where it should correctly decode! Thus you should try with and without `- erase_count` to update L on your own implementation and see which one works OK without producing wrong decoding failures.

                # Update with the discrepancy
                err_loc = gf_poly_add(err_loc, gf_poly_scale(old_loc, delta))

        # Check if the result is correct, that there's not too many errors to correct
        while len(err_loc) and err_loc[0] == 0: del err_loc[0] # drop leading 0s, else errs will not be of the correct size
        errs = len(err_loc) - 1
        if (errs-erase_count) * 2 + erase_count > nsym:
            raise ReedSolomonError("Too many errors to correct")    # too many errors to correct

        return err_loc

    def rs_find_errata_locator(e_pos, generator=2):
        '''Compute the erasures/errors/errata locator polynomial from the erasures/errors/errata positions
        (the positions must be relative to the x coefficient, eg: "hello worldxxxxxxxxx" is tampered to "h_ll_ worldxxxxxxxxx"
        with xxxxxxxxx being the ecc of length n-k=9, here the string positions are [1, 4], but the coefficients are reversed
        since the ecc characters are placed as the first coefficients of the polynomial, thus the coefficients of the
        erased characters are n-1 - [1, 4] = [18, 15] = erasures_loc to be specified as an argument.'''

        e_loc = [1] # just to init because we will multiply, so it must be 1 so that the multiplication starts correctly without nulling any term
        # erasures_loc = product(1 - x*alpha**i) for i in erasures_pos and where alpha is the alpha chosen to evaluate polynomials.
        for i in e_pos:
            e_loc = gf_poly_mul( e_loc, gf_poly_add([1], [gf_pow(generator, i), 0]) )
        return e_loc

    def rs_find_error_evaluator(synd, err_loc, nsym):
        '''Compute the error (or erasures if you supply sigma=erasures locator polynomial, or errata) evaluator polynomial Omega
        from the syndrome and the error/erasures/errata locator Sigma.'''

        # Omega(x) = [ Synd(x) * Error_loc(x) ] mod x^(n-k+1)
        _, remainder = gf_poly_div( gf_poly_mul(synd, err_loc), ([1] + [0]*(nsym+1)) ) # first multiply syndromes * errata_locator, then do a
                                                                                    # polynomial division to truncate the polynomial to the
                                                                                    # required length

        # Faster way that is equivalent
        #remainder = gf_poly_mul(synd, err_loc) # first multiply the syndromes with the errata locator polynomial
        #remainder = remainder[len(remainder)-(nsym+1):] # then slice the list to truncate it (which represents the polynomial), which
                                                        # is equivalent to dividing by a polynomial of the length we want

        return remainder

    def rs_find_errors(err_loc, nmess, generator=2): # nmess is len(msg_in)
        '''Find the roots (ie, where evaluation = zero) of error polynomial by brute-force trial, this is a sort of Chien's search
        (but less efficient, Chien's search is a way to evaluate the polynomial such that each evaluation only takes constant time).'''
        errs = len(err_loc) - 1
        err_pos = []
        for i in xrange(nmess): # normally we should try all 2^8 possible values, but here we optimize to just check the interesting symbols
            if gf_poly_eval(err_loc, gf_pow(generator, i)) == 0: # It's a 0? Bingo, it's a root of the error locator polynomial,
                                                            # in other terms this is the location of an error
                err_pos.append(nmess - 1 - i)
        # Sanity check: the number of errors/errata positions found should be exactly the same as the length of the errata locator polynomial
        if len(err_pos) != errs:
            # couldn't find error locations
            raise ReedSolomonError("Too many (or few) errors found by Chien Search for the errata locator polynomial!")
        return err_pos

    def poly_scalarmul(self, A, b):
        return self.canonical([self.base_mul(a, b) for a in A])

    def poly_scalardiv(self, A, b):
        return self.canonical([self.base_div(a, b) for a in A])

    def canonical(self, A):
        for i in reversed(range(len(A))):
            if A[i] != 0:
                return A[: i + 1]
        return []

    def deg(self, A):
        return len(self.canonical(A)) - 1

    def lc(self, A):
        B = self.canonical(A)
        return B[-1]

    def expand_to_match(self, A, B):
        diff = len(A) - len(B)
        if diff > 0:
            return A, B + [0] * diff
        elif diff < 0:
            diff = abs(diff)
            return A + [0] * diff, B
        else:
            return A, B

    def poly_divmod(self, A, B):
        t = self.base_inverse(self.lc(B))
        Q = [0] * len(A)
        R = copy(A)
        for i in reversed(range(0, len(A) - len(B) + 1)):
            Q[i] = self.base_mul(t, R[i + len(B) - 1])
            for j in range(len(B)):
                R[i + j] = self.base_sub(R[i + j], self.base_mul(Q[i], B[j]))
        return self.canonical(Q), self.canonical(R)

    def poly_add(self, A, B):
        F, G = self.expand_to_match(A, B)
        return self.canonical([self.base_add(f, g) for f, g in zip(F, G)])

    def poly_sub(self, A, B):
        F, G = self.expand_to_match(A, B)
        return self.canonical([self.base_sub(f, g) for f, g in zip(F, G)])

    def poly_mul(self, A, B):
        C = [0] * (len(A) + len(B) - 1)
        for i in range(len(A)):
            for j in range(len(B)):
                C[i + j] = self.base_add(C[i + j], self.base_mul(A[i], B[j]))
        return self.canonical(C)

    def poly_eval(self, A, x):
        result = 0
        for coef in reversed(A):
            result = self.base_add(coef, self.base_mul(x, result))
        return result

    def base_add(self, a, b):
        return a ^ b

    def base_sub(self, a, b):
        return a ^ b

    def base_inverse(self, a):
        _, b, _ = self.base_egcd(a, self.field_size)
        return b if b >= 0 else b + self.field_size

    def base_mul(self, a, b):
        output = 0
        while a > 0:
            if a & 1 != 0:
                output ^= b
            a = a >> 1
            b = b << 1
            if b & 0x100 != 0:
                b ^= self.prime_poly
        return output

    def base_div(self, a, b):
        return self.base_mul(a, self.base_inverse(b))

    def base_egcd(self, a, b):
        r0, r1 = a, b
        s0, s1 = 1, 0
        t0, t1 = 0, 1

        while r1 != 0:
            q, r2 = divmod(r0, r1)
            r0, s0, t0, r1, s1, t1 = r1, s1, t1, r2, s0 - s1 * q, t0 - t1 * q

        d = r0
        s = s0
        t = t0
        return d, s, t

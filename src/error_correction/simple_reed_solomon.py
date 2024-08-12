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
    """

    def __init__(self, n, k, power_of_2, prime_poly):
        if k > n:
            raise Exception("k has to be this relation to n, k <= n")
        if n > 255:
            raise Exception(
                "n has to be n <= 2^(power_of_2) to be valid"
            )
        self.field_size = 2**power_of_2
        self.block_byte_size = self.field_size // 8
        self.prime_poly = prime_poly
        self.n = n
        self.k = k
        self.t = n - k

        self.POINTS = [p for p in range(1, self.n + 1)]

    def encode(self, secret):
        polynomial = [secret] + [random.randrange(self.PRIME) for _ in range(self.T)]
        shares = [
            int_to_bytes(self.poly_eval(polynomial, p), self.size) for p in self.POINTS
        ]
        return int_to_bytes(secret, self.size), shares

    def decode(self, shares):
        # filter missing shares // Not needed for our case since we have all keys but not all are correct
        points_values = [
            (p, bytes_to_int(v)) for p, v in zip(self.POINTS, shares) if v is not None
        ]

        # decode remaining faulty
        points, values = zip(*points_values)
        polynomial, error_locator = self.gao_decoding(
            points, values, self.R, self.MAX_MANIPULATED
        )

        # check if recovery was possible
        if polynomial is None:
            raise Exception("Too many errors, cannot reconstruct")

        # recover secret
        secret = self.poly_eval(polynomial, 0)

        # possible to find faulty indicies but we dont want that
        # error_indices = [i for i, v in enumerate(self.poly_eval(error_locator, p) for p in self.POINTS) if v == 0]
        return int_to_bytes(secret, self.size)  # , error_indices

    def gao_decoding(self, points, values, max_degree, max_error_count):
        """
        Gao's Reed Solomon
        """

        # interpolate faulty polynomial
        H = self.lagrange_interpolation(points, values)

        # compute f
        F = [1]
        for xi in points:
            Fi = [self.base_sub(0, xi), 1]
            F = self.poly_mul(F, Fi)

        # run EEA-like algorithm on (F,H) to find EEA triple
        R0, R1 = F, H
        S0, S1 = [1], []
        T0, T1 = [], [1]
        while True:
            Q, R2 = self.poly_divmod(R0, R1)

            if self.deg(R0) < max_degree + max_error_count:
                G, leftover = self.poly_divmod(R0, T0)
                if leftover == []:
                    decoded_polynomial = G
                    error_locator = T0
                    return decoded_polynomial, error_locator
                else:
                    return G, T0

            R0, S0, T0, R1, S1, T1 = (
                R1,
                S1,
                T1,
                R2,
                self.poly_sub(S0, self.poly_mul(S1, Q)),
                self.poly_sub(T0, self.poly_mul(T1, Q)),
            )

    def lagrange_interpolation(self, xs, ys):
        ls = self.lagrange_polynomials(xs)
        poly = []
        for i in range(len(ys)):
            term = self.poly_scalarmul(ls[i], ys[i])
            poly = self.poly_add(poly, term)
        return poly

    def lagrange_polynomials(self, xs):
        polys = []
        for i, xi in enumerate(xs):
            numerator = [1]
            denominator = 1
            for j, xj in enumerate(xs):
                if i == j:
                    continue
                numerator = self.poly_mul(numerator, [self.base_sub(0, xj), 1])
                denominator = self.base_mul(denominator, self.base_sub(xi, xj))
            poly = self.poly_scalardiv(numerator, denominator)
            polys.append(poly)
        return polys

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

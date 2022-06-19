import math
from math import sqrt
import struct
from mult import *
import random

class Complex(object):

    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag
        # Formats our results
        #print(self.real + self.imag)

    def __add__(self, other):
        #print('\nSum:')
        return Complex(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other):
        #print('\nDifference:')
        return Complex(self.real - other.real, self.imag - other.imag)
    
    #bits, = struct.unpack('!I', struct.pack('!f', num))
    
    @staticmethod
    def float_to_bin(num):
        bits, = struct.unpack('!I', struct.pack('!f', num))
        return "{:032b}".format(bits)

    @staticmethod
    def multUtil(a, b):
        aBits = Complex.float_to_bin(a)
        bBits = Complex.float_to_bin(b)

        sign = 1 if (aBits[0]==bBits[0]) else -1
        result = Radix4PartialProductMatrix('1' + aBits[9:32],'1' + bBits[9:32], 28, 35, 'approx', 'approx', 0).getResult()
        
        exponent = math.pow(2, int(aBits[1:9], 2) + int(bBits[1:9], 2) - 300)

        return sign*result*exponent

    def __mul__(self, other):
        #print('\nProduct:')

        return Complex(Complex.multUtil(self.real, other.real) - Complex.multUtil(self.imag, other.imag),
            (Complex.multUtil(self.imag, other.real) + Complex.multUtil(self.real, other.imag)))

    def __truediv__(self, other):
        #print('\nQuotient:')
        r = (other.real**2 + other.imag**2)
        return Complex((self.real*other.real - self.imag*other.imag)/r,
            (self.imag*other.real + self.real*other.imag)/r)

    def __abs__(self):
        #print('\nAbsolute Value:')
        new = (self.real**2 + (self.imag**2)*-1)
        return Complex(sqrt(new.real))

    def __str__(self):
        return str(self.real) + " + " + str(self.imag) + "i"

def float_to_bin(num):
        bits, = struct.unpack('!I', struct.pack('!f', num))
        return "{:032b}".format(bits)

def multUtil(a, b):
        aBits = float_to_bin(a)
        bBits = float_to_bin(b)

        sign = 1 if (aBits[0]==bBits[0]) else -1
        result = Radix4PartialProductMatrix('1' + aBits[9:32],'1' + bBits[9:32], 28, 35, 'approx', 'approx', 0).getResult()
        
        exponent = math.pow(2, int(aBits[1:9], 2) + int(bBits[1:9], 2) - 300)

        return sign*result*exponent

def mult(a, b):
    #rand1 = random.uniform(0.99, 1.0)
    #rand2 = random.uniform(0.99, 1.0)
    return complex(multUtil(a.real,b.real) - multUtil(a.imag,b.imag), multUtil(a.imag,b.real) + multUtil(a.real,b.imag))


if __name__ == "__main__":
    a = 2+4j
    b = 1.9+3j
    print(mult(a,b))

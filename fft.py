#################################################################################
#
# This is FFT library implemented in python.
# The Algorithm used is FFT, decimation in time, radix -2 algorithm
# can compute FFT of 1-d and 2-d lists, 1-d for 1-d signals , and 2-d for images
#
# by : Umanga Bista @ bistaumanga@gmail.com
# 
#################################################################################

import cmath
import numpy as np
from math import log, ceil
from Complex import *

def omega(p, q):
   ''' The omega term in DFT and IDFT formulas'''
   return cmath.exp((2.0 * cmath.pi * 1j * q) / p)

def pad(lst):
   '''padding the list to next nearest power of 2 as FFT implemented is radix 2'''
   k = 0
   while 2**k < len(lst):
      k += 1
   return np.concatenate((lst, ([0] * (2 ** k - len(lst)))))

def fft(x):
   ''' FFT of 1-d signals
   usage : X = fft(x)
   where input x = list containing sequences of a discrete time signals
   and output X = dft of x '''

   n = len(x)
   if n == 1:
      return x
   Feven, Fodd = fft(x[0::2]), fft(x[1::2])
   combined = [0] * n
   for m in range(int(n/2)):
     combined[m] = Feven[m] + mult(omega(n, -m), Fodd[m])
     combined[m + int(n/2)] = Feven[m] - mult(omega(n, -m), Fodd[m])
   return np.array(combined)

def ifft(X):
   ''' IFFT of 1-d signals
   usage x = ifft(X) 
   unpadding must be done implicitly'''

   x = fft([x.conjugate() for x in X])
   return [x.conjugate()/len(X) for x in x]

def pad2(x):
   m, n = np.shape(x)
   M, N = 2 ** int(ceil(log(m, 2))), 2 ** int(ceil(log(n, 2)))
   F = np.zeros((M,N), dtype = x.dtype)
   F[0:m, 0:n] = x
   return F, m, n

def fft2(f):
   '''FFT of 2-d signals/images with padding
   usage X, m, n = fft2(x), where m and n are dimensions of original signal'''

   f, m, n = pad2(f)
   return np.transpose(fft(np.transpose(fft(f)))), m, n

def ifft2(F, m, n):
   ''' IFFT of 2-d signals
   usage x = ifft2(X, m, n) with unpaded, 
   where m and n are odimensions of original signal before padding'''

   f, M, N = fft2(np.conj(F))
   f = np.matrix(np.real(np.conj(f)))/(M*N)
   return f[0:m, 0:n]

def fftshift(F):
   ''' this shifts the centre of FFT of images/2-d signals'''
   M, N = F.shape
   R1, R2 = F[0: M/2, 0: N/2], F[M/2: M, 0: N/2]
   R3, R4 = F[0: M/2, N/2: N], F[M/2: M, N/2: N]
   sF = np.zeros(F.shape,dtype = F.dtype)
   sF[M/2: M, N/2: N], sF[0: M/2, 0: N/2] = R1, R4
   sF[M/2: M, 0: N/2], sF[0: M/2, N/2: N]= R3, R2
   return sF
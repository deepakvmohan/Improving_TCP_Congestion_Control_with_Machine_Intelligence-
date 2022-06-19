from __future__ import division
import numpy as np
#from scikits.bootstrap import flacread
from scipy.io.wavfile import read
from numpy.fft import rfft, irfft
from numpy import argmax, sqrt, mean, absolute, linspace, log10, logical_and, average, diff, correlate
from scipy.signal import blackmanharris, fftconvolve
import time
import sys
from fft import fft
# from signaltoolsmod import fftconvolve
from parabolic import parabolic

def parabolic(f, x):
    xv = 0.5 * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 0.25 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)

    
def freq_from_fft(sig, fs):
    windowed = sig * np.blackman(len(sig))
    #print(type(windowed))
    f = rfft(windowed)
    #print(type(f))
    
    i = argmax(abs(f)) 
    true_i = parabolic(abs(f), i)[0]
    
    # Convert to equivalent frequency
    return fs * true_i / len(windowed)


filename = sys.argv[1]

print ('Reading file "%s"\n' % filename)
a = read(filename)
fs = a[0]
signal = np.array(a[1],dtype=float)
#print(signal[0])
print ('Calculating frequency from FFT:')
start_time = time.time()
print ('%f Hz'   % freq_from_fft(signal, fs))




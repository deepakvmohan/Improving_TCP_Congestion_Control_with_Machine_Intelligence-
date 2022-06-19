from fft import *
import numpy as np
from numpy.fft import rfft, irfft

print ('Testing for 1-d signals')
# Generating sin curve in range [-2p, 2pi] with 128 sample points
f = np.sin(np.linspace(-2*np.pi,2*np.pi,16))
# let us add some noise with mean 0.5 and sigma 0.75
f = f + 0.75 * np.random.rand(16) + 0.5
F = fft(f)
F2 = rfft(f)
print(F)
print(F2)
"""
import pylab as plt
fig = plt.figure()
fig.add_subplot(311)
plt.plot(f)
plt.title('Original Signal')
fig.add_subplot(312)
plt.plot(np.log(np.abs(F[:64]) + 1))
plt.title('magnitude plot')
fig.add_subplot(313)
plt.plot(np.angle(F[:64]))
plt.title('Phase plot')
plt.show()"""


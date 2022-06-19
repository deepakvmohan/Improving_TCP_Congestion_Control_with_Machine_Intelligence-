from mult import *
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

res = []
res.append([ "multiplicand", "multiplier", "accurate result", "approximate result", "accuracy"])
acc = []

for _ in range(100000):
	temp = []
	multiplier = random.randint(0, 2**23 - 1)
	multiplicand = random.randint(0, 2**23 - 1)
	approx = Radix4PartialProductMatrix('{0:b}'.format(multiplier),'{0:b}'.format(multiplicand), 24, 10, "acc", "approx")
	#acc =  Radix4PartialProductMatrix('{0:b}'.format(multiplier),'{0:b}'.format(multiplicand), 24, 0, "acc", "acc")
	#re = acc.getResult()
	rc = multiplicand*multiplier
	#rc = multiplicand*multiplier
	re = approx.getResult()
	Aamp = 1 - abs(rc-re)/re
	#acc.append(Aamp)
	#Ainf = Bit.accInf(approx.result, acc.result, 96)
	temp.append(multiplicand)
	temp.append(multiplier)
	temp.append(rc)
	temp.append(re)
	temp.append(Aamp)
	#temp.append(Ainf)
	res.append(temp)

pd.DataFrame(np.array(res)).to_csv("results/file.csv")

"""n = [0]*6

for x in acc:
	print(x)
	if x > 0.95:
		n[0]+=1
	elif x > 0.90 and x <= 0.95:
		n[1]+=1
	elif x > 0.80 and x <= 0.9:
		n[2]+=1
	elif x > 0.7 and x <= 0.8:
		n[3]+=1
	elif x > 0.6 and x <= 0.7:
		n[4]+=1

fig = plt.figure(figsize=(10, 7))
plt.yscale('log')
plt.bar(['(9.5,1.0]', '(9.0,9.5]', '(8.0,9.0]', '(7.0,8.0]', '(6.0,7.0]', '<6.0'], n)
plt.show()"""


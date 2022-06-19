import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt

df = read_csv('results/file.csv')
#data = df.values
acc = np.array(df.accuracy)

print(acc)

n = [0]*7

for x in acc:
	#print(x)
	if x >= 0.95:
		n[6]+=1
	elif x > 0.90 and x < 0.95:
		n[5]+=1
	elif x > 0.80 and x <= 0.9:
		n[4]+=1
	elif x > 0.7 and x <= 0.8:
		n[3]+=1
	elif x > 0.6 and x <= 0.7:
		n[2]+=1
	elif x > 0.5 and x <= 0.6:
		n[1]+=1
	elif x < 0.5:
		n[0]+=1

fig = plt.figure(figsize=(10, 7))
#plt.yscale('log')
x = [ '<0.5', '(0.5, 0.6]','(0.60,0.70]', '(0.70,0.80]',  '(0.80,0.90]', '(.90,0.95]', '(0.95,1.0]']
plt.barh(x, n)

for index, value in enumerate(n):
	plt.text(value, index, str(value))
plt.show()



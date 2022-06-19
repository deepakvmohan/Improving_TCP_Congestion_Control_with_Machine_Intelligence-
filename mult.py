from argparse import ArgumentParser
from itertools import tee, islice, chain
import sys

class Iterator:
	@staticmethod
	def previous_and_next(iterable):
		prevs, items, nexts = tee(iterable, 3)
		prevs = chain([None], prevs)
		nexts = chain(islice(nexts, 1, None), [None])
		return zip(prevs, items, nexts)
		
class Radix4PartialProductMatrix:
	class Radix4Encoding:
		def __iter__(self):
			self.index = -1
			return self

		def __next__(self):
			self.index += 1
			if self.index < self.n:
				res = (self.neg[self.index], self.two[self.index], self.zero[self.index])
				return res
			else:
				raise StopIteration

		def __reversed__(self):
			return ((self.neg[i], self.two[i], self.zero[i]) for i in range(int(self.n) -1 , -1 , -1))

		def __init__(self, neg, two, zero, n):
			self.n = n
			self.neg = neg
			self.two = two
			self.zero = zero

	@staticmethod
	def encode(multiplier, n):
		neg = []
		two = []
		zero = []

		"""neg.append(Bit())
		two.append(Bit())
		zero.append(~multiplier[0])"""

		#print(type(multiplier[0]), multiplier[0])
		for pre, curr, nxt in islice(Iterator.previous_and_next(multiplier), 1, n, 2):
			if(nxt is None):
				nxt = Bit(0)

			ne = (~nxt)*pre + pre*(~curr)
			tw = (~pre)*curr*nxt + pre*(~curr)*(~nxt)
			ze = (~pre)*(~curr)*(~nxt) + nxt*curr*pre
			#print(pre, curr, nxt, ne, tw, ze)

			neg.append(ne)
			two.append(tw)
			zero.append(ze)

		#print("from encode " + str(len(neg)))
		return Radix4PartialProductMatrix.Radix4Encoding(neg, two, zero, n/2)
		#return "hello"
		


	@staticmethod
	def accuratePPgen(two, zero, neg, a1, a2):
		m = a1*(~two) + a2*two
		p = ~zero*(Bit.EXOR(m, neg))
		return p

	@staticmethod
	def approxPPgen(zero, neg, a):
		x = neg*(~a)
		y = a*(~neg)*(~zero)
		return x+y

	def __init__(self, multiplier, multiplicand, n, m, pp_type = "acc", red_type = "acc", verbosity = 0):
		
		assert n > 0, "number of bits can only be a positive integer"

		multiplier = multiplier.zfill(n)
		multiplicand = multiplicand.zfill(n)
		if len(multiplier) > n:
			raise InputError(multiplier)

		if len(multiplicand) > n:
			raise InputError(multiplicand)

		self.multiplier = []
		
		for b in multiplier:
			t = ord(b) - ord('0')
			bit = Bit(t)
			self.multiplier.append(bit)

		self.multiplicand = []

		for b in multiplicand:
			t = ord(b) - ord('0')
			bit = Bit(t)
			self.multiplicand.append(bit)

		#self.precision = precision
		self.n = n
		self.matrix = []

		#m = int(precision*2*n)
		

		self.multEncoding = Radix4PartialProductMatrix.encode(self.multiplier, n)
		self.m = m
		self.verbosity = verbosity

		row = 0
		s = self.n
		end = 2*self.n - 1
		sign = None
		signIndex = 2*self.n - 1

		pindex = 0
		for neg, two, zero in reversed(self.multEncoding):

			i = 1
			temp = []
			
			for x in range(s - 2*row - 1):
				temp.append(None)

			firstP = Radix4PartialProductMatrix.accuratePPgen(two, zero, neg, Bit(), self.multiplicand[0])
			temp.append(firstP)

			for _, curr, nxt in Iterator.previous_and_next(self.multiplicand):
				if(nxt is None):
					nxt = Bit()
				#print(str(i) + str(self.n - m), end = " ")
				if pp_type == "acc":
					p = Radix4PartialProductMatrix.accuratePPgen(two,zero,neg,curr,nxt)
				
				else:
					if(i > self.n-m):
						p = Radix4PartialProductMatrix.approxPPgen(zero, neg, curr)
					else:
						p = Radix4PartialProductMatrix.accuratePPgen(two, zero, neg, curr, nxt)

				#p = Radix4PartialProductMatrix.approxPPgen(zero, neg, curr)

				temp.append(p)
				i+=1

			pindex+=1
			
			m = m - 2

			for x in range(self.n - s + 2*row):
				temp.append(None)

			if(row>0):
				temp[signIndex] = sign
				signIndex-=2

			if neg == Bit(1):
				sign = Bit(1)
			else:
				sign = None

			row += 1
			self.matrix.append(temp)

		temp = []

		if(self.multiplier[0] == Bit()):
			for _ in range(self.n):
				temp.append(Bit())
			for _ in range(self.n):
				temp.append(None)

		else:
			for _ in range(10):
				temp.append(Bit())
			for _ in range(self.n):
				temp.append(None)

		self.matrix.append(temp)
		if(verbosity==1):
			print("Partial Product Matrix")
			Radix4PartialProductMatrix.printPPmatrix(self.matrix)
		
		self.signExtension()

		if(verbosity==1):
			print("")
			print("After sign extension:")
			Radix4PartialProductMatrix.printPPmatrix(self.matrix)

		if red_type == "acc":
			self.result = Radix4PartialProductMatrix.accReduce(self.matrix, self.n, self.verbosity)
		else:
			self.result = Radix4PartialProductMatrix.accReduce(self.approxReduce(), self.n, self.verbosity) 
		#Radix4PartialProductMatrix.printPPmatrix(self.matrix)

		#print(self.matrix[0][0])

	@staticmethod
	def printPPmatrix(matrix):
		"""for el in range(0, len(self.multiplier)):
			print("  ", end=" ")

		for b in self.multiplier:
			print(b, end=" ")

		print("")

		for el in range(0, len(self.multiplier)):
			print("  ", end=" ")

		for b in self.multiplicand:
			print(b, end=" ")

		print("")

		for _ in range(len(self.multiplicand)):
			print("  ", end = " ")
		for _ in range(len(self.multiplicand)):
			print("--", end="-")

		print("")

		for pp, i in zip(self.matrix, range(0, self.n)):
			for j in range(self.n-2*i, 0, -1):
				print("  ", end= ' ')
			for b in pp:
				print(b, end=' ')
			print(" ")
			#print(" ")"""


		for pp in matrix:
			for b in pp:
				if b is None:
					print("  ", end = " ")
				else:
					print(b, end=" ")

			print("")

	def signExtension(self):
		index = int(self.n)
		for pp, row in zip(self.matrix, range(len(self.matrix)-1)):
			index-=2
			#print(index)
			if(row == 0):
				pp[index] = pp[index+1]
				pp[index-1] = pp[index]
				pp[index-2] = ~pp[index]
			else:
				pp[index] = ~pp[index+1]
				if(index!=0):
					pp[index-1] = Bit(1) 

	
	@staticmethod
	def accReduce(matrix, n, verbosity):
		res = []
		for _ in range(2*n):
			res.append(Bit())
			
		
		#print(type(res[0]))
		for pp in matrix:
			carry = Bit(0)
			for i in range(2*n-1, -1, -1):
				carry, res[i] = Bit.fullAdder(pp[i], res[i], carry)

		if(verbosity==1):
			#print("Accurate reduction")
			print("")
			for x in res:
				print(x, end=" ")
			print("")
		return res


	@staticmethod
	def countValidBits(mat):
		countMat = [0]*(len(mat[0]))
		#print(len(self.matrix[0]))
		for j in range(len(mat[0])):
			for i in range(len(mat)):
				if(mat[i][j] is not None):
					countMat[j]+=1
				#print(i, j)

		return countMat


	@staticmethod
	def chooseCompressor(curr, bottleneck):
		comp = 0
		if(curr-3>=bottleneck): 
			comp = 4
		elif(curr-2>=bottleneck):
			comp = 3
		else:
			comp = 2

		return comp, curr - comp, bottleneck - 1

	@staticmethod
	def reduceUtil(mat, bottleneck):
		res = [[None]*len(mat[0]) for _ in range(bottleneck)]
		#Radix4PartialProductMatrix.printPPmatrix(res)

		countMat = Radix4PartialProductMatrix.countValidBits(mat)
		#print(countMat)
		carry = 0
		pk = 0

		for j in range(len(mat[0])-1, -1, -1):
			if(countMat[j] + carry<=bottleneck):
				k = carry
				i = 0
				
				while(mat[i][j] is None):
					i+=1

				x = 0
				
				while(x<countMat[j]):
					res[k][j] = mat[i][j]
					#print("8")
					i+=1
					k+=1
					x+=1

				#Radix4PartialProductMatrix.printPPmatrix(res)
				carry = 0

			else:
				currCarry = 0
				bn = bottleneck - carry
				count = countMat[j]
				k = carry
				pk = 0
				i = 0
					
				while(mat[i][j] is None):
					i+=1
				
				while(count!=bn):
					compressor, count, bn = Radix4PartialProductMatrix.chooseCompressor(count, bn)
					if compressor == 4:
						res[pk][j-1], res[k][j] = Bit.fourtwoCompressor(mat[i][j], mat[i+1][j], mat[i+2][j], mat[i+3][j])
						i += 4
					if compressor == 3:
						res[pk][j-1], res[k][j] = Bit.fullAdder(mat[i][j], mat[i+1][j], mat[i+2][j])
						i+=3
					if compressor == 2:
						res[pk][j-1], res[k][j] = Bit.halfAdder(mat[i][j], mat[i+1][j])
						i+=2

					pk+=1
					k+=1

					currCarry+=1

				while(k!=bottleneck):
					res[k][j] = mat[i][j]
					k+=1
					i+=1

				carry = currCarry

		if bottleneck is 2:
			res[0][len(mat[0])-1] = mat[0][len(mat[0])-1]

		return res


	def getResult(self):
		return int(Bit.toString(self.result), 2)

	def approxReduce(self):

		b = 2*(len(self.matrix)//4)+len(self.matrix)%4 
		res = Radix4PartialProductMatrix.reduceUtil(self.matrix, b)
		if(self.verbosity==1):
			print("step 1")
			Radix4PartialProductMatrix.printPPmatrix(res)
		i = 1

		while(len(res)!=2):
			i+=1
			bottleneck = 2 if len(res)==3 else 2*(len(res)//4)+len(res)%4 
			res = Radix4PartialProductMatrix.reduceUtil(res, bottleneck)
			if(self.verbosity==1):
				print("step " + str(i))
				Radix4PartialProductMatrix.printPPmatrix(res)

		return res

class InputError(Exception):
	def __init__(self, inp, msg = "input not in the range"):
		self.input = inp
		self.msg = msg
		super().__init__(self.msg)

	def __str__(self):
		return f"{self.input} --> {self.msg}"

class Bit:
	def __init__(self, x=0):
		self.x = x


	def __add__(self, other):
		if(self.x is 0 and other.x is 0):
			return Bit(0)
		return Bit(1)

	def __invert__(self):
		if self.x is 0 or self.x is '0':
			return Bit(1)
		return Bit(0)

	def __mul__(self, other):
		if(self.x is 1 and other.x is 1):
			return Bit(1)
		return Bit(0)

	def __str__(self):
		return str(self.x) + " "

	def __eq__(self, other):
		return self.x == other.x

	def __ne__(self, other):
		return self.x != other.x

	@staticmethod
	def accInf(a, b, n):
		c = 0
		for x, y in zip(a, b):
			if x != y:
				c+=1
		return 1-c/n

	@staticmethod 
	def toString(arr):
		s = ""
		for n in arr:
			s += str(n.x)

		return s

	@staticmethod
	def fullAdder(a, b, carry):
		if(a is None):
			a = Bit()
		if(b is None):
			b = Bit()
		if(carry is None):
			carry = Bit()
		res = a.x + b.x + carry.x
		s = res%2
		res = res//2
		c = res%2

		return (Bit(c), Bit(s))


	@staticmethod
	def halfAdder(a, b):
		if a is None:
			a = Bit()
		if b is None:
			b = Bit()

		s = Bit.EXOR(a, b)
		c = a * b

		return c, s

	@staticmethod
	def fourtwoCompressor(a, b, c, d):
		w1 = a + b
		w2 = c + a*b
		w3 = d + a*b*c

		return Bit.fullAdder(w1, w2, w3)

	@staticmethod
	def OR(a, b):
		return a + b

	@staticmethod
	def AND(a, b):
		return a * b

	@staticmethod
	def NOT(a):
		return ~a 

	@staticmethod
	def EXOR(a, b):
		x = ~a * b
		y = a * ~b
		return x + y

	@staticmethod
	def EXNOR(a, b):
		x = a * b
		y = ~a * ~b
		return x + y

	@staticmethod
	def NAND(a, b):
		x = a * b
		return ~x

	@staticmethod
	def NOR(a, b):
		x = a + b
		return ~x



def main():
	parser = ArgumentParser(description='1st argument is multiplier and 2nd ardgument is multiplicand.')
	parser.add_argument("multiplier", type = int, help="provide multiplier")
	parser.add_argument("multiplicand", type = int, help="provide multiplicand")
	parser.add_argument('-b', "--bits", type=int, help="number of bits")
	parser.add_argument('-m', "--m", type= int, help="precision if approx ppgen is chosen")
	parser.add_argument("-accPP", "--accuratePP", help="set to use accurate partial product generator", action="store_true")
	parser.add_argument("-accRed" ,"--accurateReduction", help="to use accurate reduction", action="store_true")
	parser.add_argument("-v", "--verbosity", help="1 to display intermediate steps default = 0")
	args = parser.parse_args()

	pp_type = "approx" if args.accuratePP==False else "acc"
	red_type = "approx" if args.accurateReduction==False else "acc"

	if(args.verbosity):
		verbosity = 1
	else:
		verbosity = 0

	#multiplier = "0000011111000101"""
	args = parser.parse_args()
	multiplier = '{0:b}'.format(int(args.multiplier))
	multiplicand = '{0:b}'.format(int(args.multiplicand))
	
	m = args.m if(args.m is not None) else 0
	#approx = Radix4PartialProductMatrix(multiplier,multiplicand, 48, 10, "approx", "approx")
	#Radix4PartialProductMatrix.printPPmatrix(x.matrix)
	a = Radix4PartialProductMatrix(multiplier, multiplicand, 24, m, pp_type, red_type, verbosity)

	rc = args.multiplier*args.multiplicand
	re = a.getResult()
	print("result =", re)
	print("accuracy =", 1-abs(rc-re)/rc)



if __name__ == '__main__':
	main()
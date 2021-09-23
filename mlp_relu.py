import numpy as np
import sys

class MLPReLU:

	def __init__(self, trl, trin, tstl, tstin, first, second, output):
		self.numtr, self.numin = np.shape(trin)
		self.numtst, compare = np.shape(tstin)
		if self.numin != compare:
			sys.exit("The number of inputs for your test data vs. training data", \
				"does not match up.  Please try again.")

		self.trl = trl
		self.tstl = tstl

		self.trin = np.concatenate((np.ones((self.numtr, 1)), trin), axis=1) 
		self.tstin = np.concatenate((np.ones((self.numtst, 1)), tstin), axis=1)

		# sets the weights between levels (including that for bias node)
		self.firstweights = np.random.rand(self.numin + 1, first) / 10 - .05
		self.secndweights = np.random.rand(first + 1, second) / 10 - .05
		self.thirdweights = np.random.rand(second + 1, output) / 10 - .05
			
		print(f"We have {self.numtr} training examples with {self.numin} inputs", \
			f"each.\n{self.numtst} examples to test on.\n{first + 1}", \
				f"and {second + 1} nodes in the middle for {output} outputs.")

	def add_one(self, x):
		return x+1

	def psome(self):
		print("I am a class function!!!...")
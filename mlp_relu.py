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

		# I don't know if I actually need this?...
		# self.hidden1 = first
		# self.hidden2 = second
		# self.output = output

		# so we already add in the bias to the inputs here.
		self.trin = np.concatenate((np.ones((self.numtr, 1)), trin), axis=1) 
		self.tstin = np.concatenate((np.ones((self.numtst, 1)), tstin), axis=1)

		# sets the weights between levels (including that for bias node)
		self.firstweights = np.random.rand(self.numin + 1, first) / 10 - .05
		self.secndweights = np.random.rand(first + 1, second) / 10 - .05
		self.thirdweights = np.random.rand(second + 1, output) / 10 - .05
			
		# print statement so I know everything is working right... will delete later.
		print(f"We have {self.numtr} training examples with {self.numin} inputs", \
			f"each.\n{self.numtst} examples to test on.\n{first + 1}", \
				f"and {second + 1} nodes in the middle for {output} outputs.")

	def forward_propagation(self, row, label):
		# I need some activation function here I think, between the rows.
		hidden1 = np.concatenate(([1], np.dot(row, self.firstweights)))
		hidden1 = np.where(hidden1 < 0, 0, hidden1)
		print("Hidden1: n/", hidden1)

		hidden2 = np.concatenate(([1], np.dot(hidden1, self.secndweights)))
		hidden2 = np.where(hidden2 < 0, 0, hidden2)
		print("Hidden2: \n", hidden2)

		output = np.dot(hidden2, self.thirdweights)
		output = np.where(output < 0, 0, output)
		print("Output: \n", output)
		print("Label: ", label)

	def testing(self):
		self.forward_propagation(self.tstin[0], self.tstl[0])


	def add_one(self, x):
		return x+1

	def psome(self):
		print("I am a class function!!!...")
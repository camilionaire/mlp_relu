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

		#I don't know if I actually need this?...
		self.hidden1size = first
		self.hidden2size = second
		self.outputsize = output

		self.hidden1 = []
		self.hidden2 = []
		self.output = []

		# so we already add in the bias to the inputs here.
		# taking that out and doing it in forward propagation
		self.trin = trin
		self.tstin = tstin
		# self.trin = np.concatenate((np.ones((self.numtr, 1)), trin), axis=1) 
		# self.tstin = np.concatenate((np.ones((self.numtst, 1)), tstin), axis=1)

		# sets the weights between levels (including that for bias node)
		self.firstweights = np.random.rand(self.numin + 1, first) / 10 - .05
		self.secndweights = np.random.rand(first + 1, second) / 10 - .05
		self.thirdweights = np.random.rand(second + 1, output) / 10 - .05
			
		# print statement so I know everything is working right... will delete later.
		print(f"We have {self.numtr} training examples with {self.numin} inputs", \
			f"each.\n{self.numtst} examples to test on.\n{first + 1}", \
				f"and {second + 1} nodes in the middle for {output} outputs.")

	def forward_propagation(self, row):
		# NOTE might want to change all bias nodes to .1 instead.
		bias_row = np.concatenate(([1], row))
		# I need some activation function here I think, between the rows.
		self.hidden1 = np.concatenate(([1], np.dot(bias_row, self.firstweights)))
		self.hidden1 = np.where(self.hidden1 < 0, 0, self.hidden1)
		print("Hidden1: \n", self.hidden1)

		self.hidden2 = np.concatenate(([1], np.dot(self.hidden1, self.secndweights)))
		self.hidden2 = np.where(self.hidden2 < 0, 0, self.hidden2)
		print("Hidden2: \n", self.hidden2)

		self.output = np.dot(self.hidden2, self.thirdweights)
		self.output = np.where(self.output < 0, 0, self.output)
		print("Output: \n", self.output)
		# print("Label: ", label)

		# truth = False
		# maxi = max(output)
		# for i in range(0, self.outputsize):
		# 	if output[i] == maxi:
		# 		guess = i
		# 		if i == label:
		# 			truth = True
		# return truth

	def check_if_matches(self, label):
		truth = False
		maxi = max(self.output)
		for i in range(0, self.outputsize):
			if self.output[i] == maxi:
				guess = i
				if i == label:
					truth = True
		return truth

	def forw_and_back(self):
		# loads in 1 layer at a time, affects self.hidden1/2/output
		self.forward_propagation(self.tstin[0])
		print("label: ", self.tstl[0])
		if not self.check_if_matches(self.tstl[0]):
			self.backward_propagation()

	def backward_propagation(self):
		print("This is back propagation!!! Hello!")


	def testing(self):
		self.forw_and_back()


	def add_one(self, x):
		return x+1

	def psome(self):
		print("I am a class function!!!...")
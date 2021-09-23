import numpy as np
import sys

class MLPReLU:

	def __init__(self, trl, trin, tstl, tstin, first, second, output):
		self.trl = trl
		self.trin = trin
		self.tstl = tstl
		self.tstin = tstin
		self.numtr, self.numin = np.shape(trin)
		self.numtst, compare = np.shape(tstin)

		if self.numin != compare:
			sys.exit("The number of inputs for your test data vs. training data \
				does not match up.  Please try again.")
			
		print(f"We have {self.numtr} training examples with {self.numin} inputs \
			each.  {self.numtst} examples to test on.  {first + 1}, and {second + 1} \
				 nodes in the middle for {output} outputs.")

	def print_name(self):
		print("My name is", self.name)

	def change_year(self, year):
		self.year = year
		print("{} was released in {}.".format(self.name, self.year))

	def add_one(self, x):
		return x+1

	def psome(self):
		print("I am a class function!!!...")
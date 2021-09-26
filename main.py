import numpy as np
import mlp_relu

################################################################################
# NOTE csv files must have header as we skip row 0, and must be in a folder
# titled mnist, files named: "mnist/mnist_test.csv" and "mnist/mnist_train.csv".

# this function will read in data from two files, format the information of
# and return:
# train labels and inputs and test labels and inputs
################################################################################
def mnist_csv_format():
	print("Loading test data...")
	raw_test = np.loadtxt("mnist/mnist_test.csv", delimiter=",", skiprows=1)
	# took out to save time during building
	# print("Loading training data...")
	# raw_train = np.loadtxt("mnist/mnist_train.csv", delimiter=",", skiprows=1)

	print("Formating info...")
	# took out to save time during building
	# lab_train = raw_train[:, 0]
	# in_train = raw_train[:, 1:] / 256

	lab_test = raw_test[:, 0]
	in_test = raw_test[:, 1:] / 256

	# took out training stuff to save time during building
	return lab_test, in_test

def main():
	# basic print statements and loading of data...
	print("Starting program...")
	# took out the trainning info to save time while building
	labtest, intest = mnist_csv_format()

	# took out the trainning info to save time while building
	mlp = mlp_relu.MLPReLU(labtest, intest, 100, 50, 10)

	mlp.testing()

if __name__ == "__main__":
	main()
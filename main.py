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
	print("Loading training data...")
	raw_train = np.loadtxt("mnist/mnist_train.csv", delimiter=",", skiprows=1)

	print("Formating info...")
	lab_train = raw_train[:, 0]
	in_train = raw_train[:, 1:]

	lab_test = raw_test[:, 0]
	in_test = raw_test[:, 1:]

	return lab_train, in_train, lab_test, in_test

def main():
	# basic print statements and loading of data...
	print("Starting program...")
	labtrain, intrain, labtest, intest = mnist_csv_format()

	print("training label shape: ", np.shape(labtrain))
	print("training input size: ", np.shape(intrain))
	print("testing label shape: ", np.shape(labtest))
	print("testing input size: ", np.shape(intest))
	
	mlp = mlp_relu.MLPReLU(labtrain, intrain, labtest, intest, 100, 50, 10)

if __name__ == "__main__":
	main()
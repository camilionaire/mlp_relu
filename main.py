import numpy as np
import mlp_relu

def main():
	# basic print statements and loading of data...
	print("Starting program...")
	# print("Loading test data...")
	# raw_test = np.loadtxt("csv/mnist_test.csv", delimiter=",", skiprows=1)
	# print("Loading training data...")
	# raw_train = np.loadtxt("csv/mnist_train.csv", delimiter=",", skiprows=1)

	mlp = mlp_relu.MLPReLU()

	mlp.psome()
	mlp.print_name()
	print(mlp.add_one(5))
	mlp.change_year(1996)

if __name__ == "__main__":
	main()
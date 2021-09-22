

class MLPReLU():

	def __init__(self):
		self.name = "Happy Gill"
		self.year = 2021

	def print_name(self):
		print("My name is", self.name)

	def change_year(self, year):
		self.year = year
		print("{} was released in {}.".format(self.name, self.year))

	def add_one(self, x):
		return x+1

	def psome(self):
		print("I am a class function!!!...")
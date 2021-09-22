
class Person:
	# class attribute, no access to instance ot class.
	number_of_people = 0

	def __init__(self, name):
		self.name = name
		Person.add_person()
		self.number = Person.number_of_people

	# act on the class itself.
	@classmethod
	def number_of_people_(cls):
		return cls.number_of_people

	@classmethod
	def add_person(cls):
		cls.number_of_people += 1

p1 = Person("Tim")
p2 = Person("Jill")

print(Person.number_of_people)
Person.number_of_people = 8
print(p2.number_of_people)

print(Person.number_of_people_())

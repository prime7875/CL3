# client.py
import Pyro4

uri = input("Enter the uri of the StringConcatenator object: ")
concatenator = Pyro4.Proxy(uri)

str1 = input("Enter the first string: ")
str2 = input("Enter the second string: ")

result = concatenator.concatenate_strings(str1, str2)
print("Concatenated string:", result)

import sys
from collections import defaultdict

def mapper(text):
    for char in text:
        if char.strip():  # Ignore pure whitespace
            yield (char, 1)

def reducer(mapped_values):
    reduction = defaultdict(int)
    for key, value in mapped_values:
        reduction[key] += value
    return reduction

def charCount():
    if len(sys.argv) != 2:
        print("Usage: python local_character_count.py <file>")
        sys.exit(1)

    filename = sys.argv[1]
    with open(filename, 'r') as file:
        text = file.read()

    mapped_values = list(mapper(text))
    reduced_values = reducer(mapped_values)

    for char, count in sorted(reduced_values.items()):
        print(f'{char}: {count}')

charCount()

from collections import Counter
import csv

# see the label list
labels = list()
with open('test.csv', 'r') as istream:
    reader = csv.reader(istream, delimiter=',')
    for line in reader:
        labels.append(line[0])
c = Counter(labels)
print(c)
print(sorted(c.keys()))

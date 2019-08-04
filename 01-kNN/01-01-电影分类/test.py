import kNN
group, labels = kNN.createDataSet()

print(group)
print(labels)

print(kNN.classify0([1.9, 3], group, labels, 3))

#%%
import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# %%
training_samples = np.loadtxt("training_samples.txt")
training_labels = np.loadtxt("training_labels.txt", dtype=int)
testing_samples = np.loadtxt("testing_samples.txt")

# %%
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, data, labels):

        # Standarizing the data
        self.means = np.mean(data, axis=0)
        self.sds = np.std(data, axis=0)
        self.data = (data - self.means) / self.sds

        self.labels = labels

    def predict(self, data):
        output = []

        # Standardizing the prediction data
        data = (data - self.means) / self.sds
        for row in data:
            h = []
            for training_row, training_label in zip(self.data, self.labels):
                dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(row, training_row)))
                heapq.heappush(h, (dist, training_label))

            c = Counter(heapq.heappop(h)[1] for _ in range(self.k))
            output.append(c.most_common(1)[0][0])
        return output


rng = np.random.default_rng()
accuracy = []
for k in range(1, 200, 10):
    classifier = KNN(k)
    split_indices = np.array_split(np.arange(len(training_samples)), 5)
    acc = []
    for i in split_indices:
        val_X = training_samples[i]
        val_y = training_labels[i]

        mask = np.ones(len(training_samples), bool)
        mask[i] = False
        train_X = training_samples[mask]
        train_y = training_labels[mask]
        classifier.fit(train_X, train_y)
        acc.append(sum(classifier.predict(val_X) == val_y) / len(val_X))
    accuracy.append(np.average(acc))

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(1, 200, 10), accuracy)
plt.show()

# %%

classifier = KNN(30)
classifier.fit(training_samples, training_labels)
output = classifier.predict(testing_samples)
print(output)
print(sorted([(x, y) for x, y in Counter(output).items()], key=lambda x: x[0]))

#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
points = np.loadtxt("2D_points.txt")
x, y = points.T

fig, ax = plt.subplots()
plt.scatter(x, y)
plt.show()

#%%
X = np.vstack((np.ones(len(x)), x)).T
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
print(beta)

Y = X.dot(beta)

fig, ax = plt.subplots()
plt.scatter(x, y)
plt.plot(x, Y)
plt.show()

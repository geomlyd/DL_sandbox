import ExampleGraphs
from Optimizers import GradientDescentOptimizer
import numpy as np
import matplotlib.pyplot as plt
import pytorchFullyConnectedRegression
import pytorch_lightning as pl
import torch

opt = GradientDescentOptimizer(0.0001)
G = ExampleGraphs.FullyConnectedClassifier([[2, 50], [50, 50], [50, 2]], opt)

class1Data = 6*np.random.random(50) - 3
class1Data = np.column_stack((class1Data, class1Data*class1Data + 0.1*np.random.random(50)))

class2Data = 6*np.random.random(50) - 3
class2Data = np.column_stack((class2Data, class2Data*class2Data - 2 + 0.1*np.random.random(50)))

allData = np.vstack((class1Data, class2Data))
allClasses = np.concatenate((np.zeros(class1Data.shape[0]), np.ones(class2Data.shape[0])))
G.fit(allData, allClasses, 200)

plt.figure()
plt.scatter(class1Data[:, 0], class1Data[:, 1], marker="x")
plt.scatter(class2Data[:, 0], class2Data[:, 1], marker="o")

x, y = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-2, 9, 0.1))
points = np.vstack((x.ravel(), y.ravel())).T
cl = G(points)
cl1points = points[cl == 0, :]
cl2points = points[cl == 1, :]
plt.scatter(cl1points[:, 0], cl1points[:, 1], marker="x")
plt.scatter(cl2points[:, 0], cl2points[:, 1], marker="o")

plt.show()


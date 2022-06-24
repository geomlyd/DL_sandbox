import ExampleGraphs
from Optimizers import GradientDescentOptimizer
import numpy as np
import matplotlib.pyplot as plt

opt = GradientDescentOptimizer(0.01)
G = ExampleGraphs.LinearRegression(1, opt)

dataX = 3*np.random.random(20)
slope = 3*np.random.random(1) - 1
intercept = 4*np.random.random(1) - 2
actualY = slope*dataX + intercept
actualY += 0.3*np.random.random(actualY.shape) - 0.1
G.fit(dataX, actualY, 200)

plt.figure()
plt.scatter(dataX, actualY)


plotX = np.arange(-5, 5, 0.2)
predictedY = G(plotX)
plt.plot(plotX, predictedY)

plt.show()
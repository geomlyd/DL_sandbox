import ExampleGraphs
from Optimizers import GradientDescentOptimizer
import numpy as np
import matplotlib.pyplot as plt

opt = GradientDescentOptimizer(0.00001)
G = ExampleGraphs.FullyConnectedRegressor([[1, 10], [10, 10], [10, 1]], opt)

dataX = np.random.random(200)
slope = 3*np.random.random(1) - 1
intercept = 0.5*np.random.random(1) - 2
actualY = dataX*dataX #slope*dataX + intercept  #dataX*dataX#
actualY += 0.3*np.random.random(actualY.shape) - 0.1
G.fit(dataX, actualY, 200)

plt.figure()
plt.scatter(dataX, actualY)


plotX = np.arange(-2, 2, 0.01)
predictedY = G(plotX)
plt.plot(plotX, predictedY)

plt.show()

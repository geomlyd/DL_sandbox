import ExampleGraphs
import numpy as np
import matplotlib.pyplot as plt

G = ExampleGraphs.InterestingGraph()

xRange = np.arange(-1.5, 1.5, 0.001)
yRange, d = G(xRange)
pos = np.random.randint(0, xRange.shape[0], size=5)

xToDiff = xRange[pos]
yVal = yRange[pos]
slopes = d[pos]

plt.figure()
plt.plot(xRange, yRange)
for i in range(pos.shape[0]):
    print(i)
    plt.scatter(xToDiff[i], yVal[i], marker='x', color='red')
    x = np.arange(xToDiff[i] - 0.2, xToDiff[i] + 0.2, 0.1)
    y = x*slopes[i] + (yVal[i] - slopes[i]*xToDiff[i])
    plt.plot(x, y, color='green')
plt.show()
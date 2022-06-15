import ExampleGraphs
import numpy as np
import matplotlib.pyplot as plt

G = ExampleGraphs.InterestingGraph()

xRange = np.arange(-0.5, 0.5, 0.001)
yRange = G(xRange)
plt.figure()
plt.plot(xRange, yRange)
plt.show()
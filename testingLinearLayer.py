import ExampleGraphs
import numpy as np
import matplotlib.pyplot as plt

G = ExampleGraphs.Rotation(90)

points = np.random.random((5, 2))
#points = np.hstack((points, points*2))
loops = 5
plt.figure()
#plt.scatter(points[:, 0], points[:, 1], marker='x')
for i in range(loops):
    G.setDegrees(i*360/loops)
    rotatedPoints = G(points)

    plt.scatter(rotatedPoints[:, 0], rotatedPoints[:, 1])
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
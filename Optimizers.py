from Optimizer import Optimizer
import numpy as np

class GradientDescentOptimizer(Optimizer):

    def __init__(self, lr):
        super().__init__(lr)

    def computeStep(self, params : np.array):
        return -self.learningRate*params
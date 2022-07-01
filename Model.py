from abc import ABC, abstractclassmethod, abstractmethod
from cmath import inf

import Optimizer
import ComputationalGraph
import numpy as np

class Model():

    def __init__(self, G : ComputationalGraph):
        self.G = G

    def fit(self, x, y, numEpochs : int, batchSize : int, o : Optimizer):

        self.G.optimizer = o

        numBatches = int(np.ceil(x.shape[0]/batchSize))
        for i in range(numEpochs):
            dataPermutation = np.random.permutation(x.shape[0])
            
            for batch in range(numBatches):
                #sampling with replacement
                batchIndices = np.random.random_integers(0, high = x.shape[0] - 1, size=batchSize)
                
                self.loadInput(x[batchIndices, ...], 
                    y[batchIndices, ...])
                self.G.runForwardPass()
                self.G.runBackwardPass()

    @abstractmethod
    def loadInput(self, *args):
        pass

    @abstractmethod           
    def __call__(self, x):
        pass


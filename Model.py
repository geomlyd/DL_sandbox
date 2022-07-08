from abc import ABC, abstractclassmethod, abstractmethod
from cmath import inf
from random import sample

import Optimizer
import ComputationalGraph
import numpy as np

class Model():

    def __init__(self, G : ComputationalGraph):
        self.G = G

    def fit(self, x, y, numEpochs : int, batchSize : int, o : Optimizer, sampleWithReplacement=False):

        self.G.optimizer = o

        numBatches = int(np.ceil(x.shape[0]/batchSize))
        for i in range(numEpochs):
            dataPermutation = np.random.permutation(x.shape[0])
            low = 0 
            for batch in range(numBatches):

                if(sampleWithReplacement):
                    batchIndices = np.random.random_integers(0, high = x.shape[0] - 1, size=batchSize)
                else:
                    batchIndices = np.arange(low, low + batchSize)
                    low += batchSize
                
                self.loadInput(x[batchIndices, ...], 
                    y[batchIndices, ...])
                self.G.runForwardPass()
                self.G.runBackwardPass()
                print(self.G.getNode("loss").value)

    @abstractmethod
    def loadInput(self, *args):
        pass

    @abstractmethod           
    def __call__(self, x):
        pass


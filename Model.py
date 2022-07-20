from abc import ABC, abstractclassmethod, abstractmethod

from typing import Tuple
import Optimizer
from ComputationalGraph import ComputationalGraph
import numpy as np
from Dataset import Dataset

class Model():

    def __init__(self, G : ComputationalGraph):
        self.G = G

    def fit(self, dataset : Dataset, numEpochs : int, batchSize : int, o : Optimizer, sampleWithReplacement=False):

        self.G.optimizer = o

        dataLen = len(dataset)
        numBatches = int(np.ceil(dataLen/batchSize))
        for i in range(numEpochs):
            dataPermutation = np.random.permutation(dataLen)
            low = 0 
            for batch in range(numBatches):

                if(sampleWithReplacement):
                    batchIndices = np.random.random_integers(0, high = dataLen - 1, size=batchSize)
                else:
                    batchIndices = np.arange(low, low + batchSize)
                    low += batchSize
                
                self.loadInput(dataset.getTrainingDataFromIndices(batchIndices))
                self.G.runForwardPass()
                self.G.runBackwardPass()
            print("Epoch: {0}, loss: {1}".format(i, self.G.getNode("loss").value))

    @abstractmethod
    def loadInput(self, input : Tuple):
        pass

    @abstractmethod           
    def __call__(self, x):
        pass


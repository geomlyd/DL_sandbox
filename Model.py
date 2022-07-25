from abc import ABC, abstractclassmethod, abstractmethod

from typing import Tuple, List, Any
import Optimizer
from ComputationalGraph import ComputationalGraph
import numpy as np
from Dataset import Dataset

class Model():

    def __init__(self, G : ComputationalGraph):
        self.G = G
        self.logDict = {}
        self.epoch = -1

    def fit(self, trainDataset : Dataset, valDataset : Dataset, 
        numEpochs : int, trainBatchSize : int, valBatchSize : int, o : Optimizer, sampleWithReplacement=False):

        self.G.optimizer = o

        trainDataLen = len(trainDataset)
        numTrainBatches = int(np.ceil(trainDataLen/trainBatchSize))

        valDataLen = len(valDataset)
        numValBatches = int(np.ceil(valDataLen/valBatchSize))
        for i in range(numEpochs):
            self.epoch = i
            dataPermutation = np.random.permutation(trainDataLen)
            low = 0 
            for batch in range(numTrainBatches):

                if(sampleWithReplacement):
                    batchIndices = np.random.random_integers(0, high = trainDataLen - 1, size=trainBatchSize)
                else:
                    batchIndices = dataPermutation[np.arange(low, 
                        min(trainDataLen - 1, low + trainBatchSize))]
                    low += trainBatchSize
                
                self.trainingStep(trainDataset.getDataFromIndices(batchIndices))

            dataPermutation = np.random.permutation(valDataLen)
            low = 0
            for batch in range(numValBatches):

                if(sampleWithReplacement):
                    batchIndices = np.random.random_integers(0, high = valDataLen - 1, size=valBatchSize)
                else:
                    batchIndices = dataPermutation[np.arange(low, 
                        min(valDataLen, low + valBatchSize))]
                    low += valBatchSize 

                self.validationStep(valDataset.getDataFromIndices(batchIndices))

            self.onEpochEnd()

    def log(self, quantityName : str, quantityValue):
        if(quantityName not in self.logDict):
            self.logDict[quantityName] = [[self.epoch, quantityValue]]
        else:
            self.logDict[quantityName].append([self.epoch, quantityValue])

    def onEpochEnd(self):
        pass

    @abstractmethod
    def trainingStep(self, trainingBatch):
        pass

    @abstractmethod
    def validationStep(self, validationBatch):
        pass

    @abstractmethod           
    def __call__(self, x):
        pass


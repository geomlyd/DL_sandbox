from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class Optimizer(ABC):

    def __init__(self, lr):
        self._learningRate = lr

    @property
    def learningRate(self):
        return self._learningRate

    @learningRate.setter
    def learningRate(self, lr):
        self._learningRate = lr

    @abstractmethod
    def computeStep(self, params : np.array):
        pass

    def computeStepWrapper(self, params : List[np.array]) -> List[np.array]:
        concatenatedParams = np.concatenate(params)
        arrayLengths = [_.shape[0] for _ in params]
        startingIndices = np.cumsum(arrayLengths)

        step = self.computeStep(concatenatedParams)

        #split the concatenated array in the indices that initial sub-array
        #sizes indicate, discard the last element since it'll be empty
        unconcatenatedParams = np.split(step, startingIndices)[:-1]
        return unconcatenatedParams


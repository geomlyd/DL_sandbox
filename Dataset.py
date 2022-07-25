from abc import ABC, abstractmethod
import numpy as np

class Dataset(ABC):

    @abstractmethod
    def __len__(self): 
        pass

    @abstractmethod
    def getDataFromIndices(self, ind : np.array):
        pass

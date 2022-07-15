



from abc import ABC, abstractmethod
import numpy as np

class Dataset(ABC):

    @abstractmethod
    def __len__(self): 
        pass

    @abstractmethod
    def getTrainingDataFromIndices(self, ind : np.array):
        pass

    @abstractmethod
    def getValidationDataFromIndices(self, ind : np.array):
        pass    

    @abstractmethod
    def getTestDataFromIndices(self, ind : np.array):
        pass    
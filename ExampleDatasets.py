from Dataset import Dataset
import numpy as np
import os

class SimpleDataset(Dataset):

    def __init__(self, x : np.array, y : np.array):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def getTrainingDataFromIndices(self, ind: np.array):
        return (self.x[ind, ...], self.y[ind, ...])

    def getValidationDataFromIndices(self, ind: np.array):
        return None

    def getTestDataFromIndices(self, ind: np.array):
        return None

class MNISTDataset(Dataset):

    def __init__(self, dirPath : str):
        if(os.path.isfile(dirPath)):
            raise FileExistsError("MNIST dataset: path is already a file, MNIST could not be downloaded")
        elif(not os.path.isdir(dirPath)):
            
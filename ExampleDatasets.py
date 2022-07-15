from tkinter import filedialog

from matplotlib import image
from Dataset import Dataset
import numpy as np
import os
import urllib.request
import gzip

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
        
        trainDataPath = os.path.join(dirPath, "trainData.gz")
        trainLabelsPath = os.path.join(dirPath, "trainLabels.gz")
        testDataPath = os.path.join(dirPath, "testData.gz")
        testLabelsPath = os.path.join(dirPath, "testLabels.gz")
        if(not os.path.isdir(dirPath)):
            os.mkdir(dirPath)
        
        urlList = ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"]
        filenameList = [trainDataPath, trainLabelsPath, testDataPath, testLabelsPath]
        for i in range(len(urlList)):
            urllib.request.urlretrieve(urlList[i], filenameList[i])
            
        if(not (os.path.isfile(trainDataPath) and os.path.isfile(trainLabelsPath) 
            and os.path.isfile(testDataPath) and os.path.isfile(trainLabelsPath))):
            raise FileNotFoundError("MNIST dataset: files were not found and could not be created in the"
                "specified directory")

        trainingData = self.processImagesGz(trainDataPath, 2051)
        import matplotlib.pyplot as plt
        plt.figure()
        print(trainingData[0, :, :])
        plt.imshow(trainingData[0, :, :])
        plt.show()


    def processImagesGz(self, path, magicNumber):

        with (gzip.GzipFile(path) as fileData):
            fileMagicNumber = int.from_bytes(fileData.read(4), "big")
            if(fileMagicNumber != magicNumber):
                print("MNIST dataset: file \"" + path + "\" is corrupted")
                exit(-1)
            
            numImages = int.from_bytes(fileData.read(4), "big")
            numRows = int.from_bytes(fileData.read(4), "big")
            numCols = int.from_bytes(fileData.read(4), "big")
            images = np.zeros((numImages, numRows, numCols), dtype=np.uint8)
            for i in range(numImages):
                pixelValues = fileData.read(numRows*numCols)
                images[i, :, :] = np.frombuffer(pixelValues, dtype=np.uint8).reshape(28, 28)

            return images
            
            


    def __len__(self):
        return None

    def getTrainingDataFromIndices(self, ind: np.array):
        return None

    def getValidationDataFromIndices(self, ind: np.array):
        return None

    def getTestDataFromIndices(self, ind: np.array):
        return None        
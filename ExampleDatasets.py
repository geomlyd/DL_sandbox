from Dataset import Dataset
import numpy as np
import os
import urllib.request
import gzip
from Transform import Transform


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

    def __init__(self, dirPath : str, train : bool = True, transform : Transform = None):
        if(os.path.isfile(dirPath)):
            raise FileExistsError("MNIST dataset: path is already a file, MNIST could not be downloaded")
        
        dataPath = os.path.join(dirPath, "trainData.gz") if train else os.path.join(dirPath, "testData.gz")
        labelsPath = os.path.join(dirPath, "trainLabels.gz") if train else os.path.join(dirPath, "testLabels.gz")

        self.transform = transform

        if(not os.path.isdir(dirPath)):
            os.mkdir(dirPath)
        
        urlList = ["http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
            if train else "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
            if train else "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"]
        filenameList = [dataPath, labelsPath]
        for i in range(len(urlList)):
            urllib.request.urlretrieve(urlList[i], filenameList[i])
            
        if(not (os.path.isfile(dataPath) and os.path.isfile(labelsPath))):
            raise FileNotFoundError("MNIST dataset: files were not found and could not be created in the"
                "specified directory")

        self.data = self.processImagesGz(dataPath, 2051)
        self.data = np.array(self.data, dtype=float)/255

        self.labels = self.processLabelsGz(labelsPath, 2049)


    def processImagesGz(self, path, magicNumber):

        with (gzip.GzipFile(path) as fileData):
            fileMagicNumber = int.from_bytes(fileData.read(4), byteorder="big")
            if(fileMagicNumber != magicNumber):
                print("MNIST dataset: file \"" + path + "\" is corrupted")
                exit(-1)
            
            numImages = int.from_bytes(fileData.read(4), byteorder="big")
            numRows = int.from_bytes(fileData.read(4), byteorder="big")
            numCols = int.from_bytes(fileData.read(4), byteorder="big")
            images = np.zeros((numImages, numRows, numCols), dtype=np.uint8)
            for i in range(numImages):
                pixelValues = fileData.read(numRows*numCols)
                images[i, :, :] = np.frombuffer(pixelValues, dtype=np.uint8).reshape(numRows, numCols)
            
            assert(np.all(images < 256))
            return images
            
    def processLabelsGz(self, path, magicNumber):

        with (gzip.GzipFile(path) as fileData):
            fileMagicNumber = int.from_bytes(fileData.read(4), byteorder="big")
            if(fileMagicNumber != magicNumber):
                print("MNIST dataset: file \"" + path + "\" is corrupted")
                exit(-1)
            
            numLabels = int.from_bytes(fileData.read(4), byteorder="big")

            labels = np.full(numLabels, 10, dtype=np.uint8)
            bytesPerRead = 700
            for i in range(int(np.ceil(numLabels/bytesPerRead))):
                pixelValues = fileData.read(min(bytesPerRead, numLabels - i*bytesPerRead))
                labels[i*bytesPerRead:i*bytesPerRead + bytesPerRead] = np.frombuffer(pixelValues, dtype=np.uint8)
            
            assert(np.all(labels < 10))
            return labels

    def __len__(self):
        return self.labels.shape[0]

    def getDataFromIndices(self, ind: np.array):
        return (self.data[ind, :, :] if self.transform is None else self.transform(self.data[ind, :, :]),
            self.labels[ind])
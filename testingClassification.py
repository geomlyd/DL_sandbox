from Optimizers import GradientDescentOptimizer
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from ExampleModels import FullyConnectedClassifier
from pytorchSimpleModels import Pytorch_FullyConnectedClassifier, Pytorch_Simple_DataModule
import torch

class1Data = np.random.random(100) - 1/2
class1Data = np.column_stack((class1Data, 2*class1Data*class1Data + 0.1*np.random.random(class1Data.shape[0])))

class2Data = np.random.random(100) - 1/2
class2Data = np.column_stack((class2Data, class2Data*class2Data - 0.15 + 0.1*np.random.random(class1Data.shape[0])))

allData = np.vstack((class1Data, class2Data))
allClasses = np.concatenate((np.zeros(class1Data.shape[0]), np.ones(class2Data.shape[0])))

whichModel = "mine"
batchSize = 100
numEpochs = 500
lr = 0.1
layerDims = [[2, 200], [200, 2]]

if(whichModel == "mine"):
    opt = GradientDescentOptimizer(lr)
    model = FullyConnectedClassifier(layerDims)
    model.fit(allData, allClasses, numEpochs, batchSize, opt)
else:
    model = Pytorch_FullyConnectedClassifier(layerDims, lr=lr)
    dataModule = Pytorch_Simple_DataModule(allData, allClasses, batchSize)
    trainer = pl.Trainer(gpus=0, max_epochs=numEpochs)
    trainer.fit(model, dataModule)

xRange = np.arange(np.min(allData[:, 0]) - 0.1, np.max(allData[:, 0]) + 0.1, 0.01)
yRange = np.arange(np.min(allData[:, 1]) - 0.1, np.max(allData[:, 1]) + 0.1, 0.01)
plotX, plotY = np.meshgrid(xRange, yRange)
plotXY = np.vstack([plotX.ravel(), plotY.ravel()])
plotXY = plotXY.T

if(not whichModel == "mine"):
    plotXY = torch.tensor(plotXY).float()

predictedClass = model(plotXY)
if(not whichModel == "mine"):
    predictedClass = predictedClass.detach().numpy()
    plotXY = plotXY.detach().numpy()
predictedClass = np.argmax(predictedClass, axis=1)
predictedClass = np.reshape(predictedClass.T, (yRange.shape[0], xRange.shape[0]))
plt.figure()
plt.contourf(xRange, yRange, predictedClass)

plt.scatter(class1Data[:, 0], class1Data[:, 1], marker='x')
plt.scatter(class2Data[:, 0], class2Data[:, 1], marker='o')
#plt.scatter(plotXY[predictedClass == 0, 0], plotXY[predictedClass == 0, 1], marker='.', color='black')
#plt.scatter(plotXY[predictedClass == 1, 0], plotXY[predictedClass == 1, 1], marker='.', color='white')
#plotX = np.arange(-1, 1, 0.01) if model == "mine" else torch.arange(-1, 1, 0.01)
#predictedY = model(plotX)
plt.show()



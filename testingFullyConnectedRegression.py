from ExampleModels import FullyConnectedRegressor
from Optimizers import GradientDescentOptimizer
import numpy as np
import matplotlib.pyplot as plt
from pytorchSimpleModels import Pytorch_FullyConnectedRegressor, Pytorch_Simple_DataModule
import pytorch_lightning as pl
import torch
import ExampleDatasets

dataDim = 1
trainingDataX = 2*np.random.random(200) - 1
trainingDataY = 1/2*(1 + np.cos(trainingDataX*2*np.pi))#slope*trainingDataX + intercept
trainingDataY += 0.01*np.random.random(trainingDataY.shape)
plt.figure()
plt.scatter(trainingDataX, trainingDataY)

whichModel = "mine"
batchSize = 2
numEpochs = 200
lr = 0.01
layerDims = [[1, 200], [200, 1]]

if(whichModel == "mine"):
    opt = GradientDescentOptimizer(lr)
    model = FullyConnectedRegressor(layerDims)
    dataset = ExampleDatasets.SimpleDataset(trainingDataX, trainingDataY)
    model.fit(dataset, numEpochs, batchSize, opt)

else:
    model = Pytorch_FullyConnectedRegressor(layerDims, lr=lr)
    dataModule = Pytorch_Simple_DataModule(trainingDataX, trainingDataY, batchSize)
    trainer = pl.Trainer(gpus=0, max_epochs=numEpochs)
    trainer.fit(model, dataModule)

plotX = np.arange(-1, 1, 0.01) if model == "mine" else torch.arange(-1, 1, 0.01)
predictedY = model(plotX)
if(not whichModel == "mine"):
    predictedY = predictedY.detach().numpy()
plt.plot(plotX, predictedY)
plt.show()
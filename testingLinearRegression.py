from ExampleModels import LinearRegression
from Optimizers import GradientDescentOptimizer
import numpy as np
import matplotlib.pyplot as plt
from pytorchSimpleModels import Pytorch_LinearRegression, Pytorch_Simple_DataModule
import pytorch_lightning as pl
import torch
import ExampleDatasets

dataDim = 1
trainingDataX = 3*np.random.random(20)
slope = 3*np.random.random(1) - 1
intercept = 4*np.random.random(1) - 2
trainingDataY = slope*trainingDataX + intercept
trainingDataY += 0.3*np.random.random(trainingDataY.shape) - 0.1
plt.figure()
plt.scatter(trainingDataX, trainingDataY)

whichModel = "mine"
batchSize = 10
numEpochs = 200
lr = 0.1

if(whichModel == "mine"):
    opt = GradientDescentOptimizer(lr)
    model = LinearRegression(dataDim)
    dataset = ExampleDatasets.SimpleDataset(trainingDataX, trainingDataY)
    model.fit(dataset, numEpochs, batchSize, opt)

else:
    model = Pytorch_LinearRegression(1, lr=lr)
    dataModule = Pytorch_Simple_DataModule(trainingDataX, trainingDataY, batchSize)
    trainer = pl.Trainer(gpus=0, max_epochs=numEpochs)
    trainer.fit(model, dataModule)

plotX = np.arange(-5, 5, 0.2) if model == "mine" else torch.arange(-5, 5, 0.2)
predictedY = model(plotX)
if(not model == "mine"):
    predictedY = predictedY.detach().numpy()
plt.plot(plotX, predictedY)
plt.show()
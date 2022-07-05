import ExampleModels
from Optimizers import GradientDescentOptimizer
import numpy as np
import matplotlib.pyplot as plt
import pytorchFullyConnectedRegression
import pytorch_lightning as pl
import torch

opt = GradientDescentOptimizer(0.001)
G = ExampleModels.FullyConnectedRegressor([[1, 10], [10, 10], [10, 1]])

dataX = np.random.random(200) - 0.5
slope = np.random.random(1)
intercept = np.random.random(1)
actualY = np.sin(dataX*dataX/0.1)#
#actualY += 0.3*np.random.random(actualY.shape)

plt.figure()
plt.scatter(dataX, actualY)

whichModel = "mine"
if(whichModel == "mine"):
    G.fit(dataX, actualY, 500, 10, opt)

    plotX = np.arange(-2, 2, 0.01)
    predictedY = G(plotX)
    plt.plot(plotX, predictedY)

    plt.show()
else:
    model = pytorchFullyConnectedRegression.Pytorch_Regressor()
    dataModule = pytorchFullyConnectedRegression.Pytorch_Regressor_DataModule(dataX, actualY)
    trainer = pl.Trainer(gpus=0, max_epochs=200)
    trainer.fit(model, dataModule)

    plotX = np.arange(-2, 2, 0.01)
    predictedY = model(torch.Tensor(plotX[:, None]))
    plt.plot(plotX, predictedY.detach().numpy())
    plt.show()

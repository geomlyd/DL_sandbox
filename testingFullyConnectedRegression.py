import ExampleGraphs
from Optimizers import GradientDescentOptimizer
import numpy as np
import matplotlib.pyplot as plt
import pytorchFullyConnectedRegression
import pytorch_lightning as pl
import torch

opt = GradientDescentOptimizer(0.001)
G = ExampleGraphs.FullyConnectedRegressor([[1, 10], [10, 10], [10, 1]], opt)

dataX = np.random.random(200) - 0.5
slope = np.random.random(1)
intercept = np.random.random(1)
actualY = dataX*dataX#
#actualY += 0.3*np.random.random(actualY.shape)

plt.figure()
plt.scatter(dataX, actualY)

model = "mine"
if(model == "mine"):
    G.fit(dataX, actualY, 500)



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

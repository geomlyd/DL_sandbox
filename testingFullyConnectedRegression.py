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
slope = 3*np.random.random(1) - 1
intercept = 0.5*np.random.random(1) - 2
actualY = dataX*dataX #slope*dataX + intercept  #dataX*dataX#
#actualY += 0.3*np.random.random(actualY.shape)

plt.figure()
plt.scatter(dataX, actualY)

model = "other"
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

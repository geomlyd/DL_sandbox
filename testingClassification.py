from Optimizers import GradientDescentOptimizer
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import ExampleModels
import pytorchFullyConnectedRegression
import torch

opt = GradientDescentOptimizer(0.001)
model = ExampleModels.FullyConnectedClassifier([[2, 50], [50, 50], [50, 2]])

class1Data = 6*np.random.random(50) - 3
class1Data = np.column_stack((class1Data, class1Data*class1Data + 0.1*np.random.random(50)))

class2Data = 6*np.random.random(50) - 3
class2Data = np.column_stack((class2Data, class2Data*class2Data - 2 + 0.1*np.random.random(50)))

allData = np.vstack((class1Data, class2Data))
allClasses = np.concatenate((np.zeros(class1Data.shape[0]), np.ones(class2Data.shape[0])))

whichModel = "mine"

plt.figure()
plt.scatter(class1Data[:, 0], class1Data[:, 1], marker="x")
plt.scatter(class2Data[:, 0], class2Data[:, 1], marker="o")
if(whichModel == "mine"):
    model.fit(allData, allClasses, 200, 10, opt)


    x, y = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-2, 9, 0.1))
    points = np.vstack((x.ravel(), y.ravel())).T
    cl = model(points)
    cl1points = points[cl == 0, :]
    cl2points = points[cl == 1, :]
    plt.scatter(cl1points[:, 0], cl1points[:, 1], marker="x")
    plt.scatter(cl2points[:, 0], cl2points[:, 1], marker="o")
else:
    model = pytorchFullyConnectedRegression.Pytorch_FullyConnected(isClassifier=True, numClasses=2)
    dataModule = pytorchFullyConnectedRegression.Pytorch_Simple_DataModule(allData, allClasses)
    trainer = pl.Trainer(gpus=0, max_epochs=500)
    trainer.fit(model, dataModule)

    x, y = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-2, 9, 0.1))
    points = np.vstack((x.ravel(), y.ravel())).T
    plotX = np.arange(-2, 2, 0.01)
    cl = model(torch.Tensor(points)).detach().numpy()
    cl = np.argmax(cl, axis=1)
    cl1points = points[cl == 0, :]
    cl2points = points[cl == 1, :]
    plt.scatter(cl1points[:, 0], cl1points[:, 1], marker="x")
    plt.scatter(cl2points[:, 0], cl2points[:, 1], marker="o")    
    

plt.show()



from matplotlib.colors import Normalize
import ExampleModels
import ExampleDatasets
import Optimizers
import Transforms


whichModel = "mine"
layerDims = [[784, 32], [32, 10]]
channelMean = 0.1307
channelStd = 0.3081
lr = 0.01
numEpochs = 100
batchSize = 128

if(whichModel == "mine"):
    opt = Optimizers.GradientDescentOptimizer(lr)
    trainingDataset = ExampleDatasets.MNISTDataset("./MNIST",
        transform=Transforms.NormalizeImage(channelMean, channelStd), train=True)
    validationDataset = ExampleDatasets.MNISTDataset("./MNIST",
        train=False)
    model = ExampleModels.FullyConnectedClassifier(layerDims)
    model.fit(trainingDataset, validationDataset, numEpochs, batchSize, 
        batchSize, opt)
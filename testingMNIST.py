from matplotlib.colors import Normalize
import ExampleModels
import ExampleDatasets
import Optimizers
import Transforms


whichModel = "mine"
layerDims = [[784, 300], [300, 100], [100, 10]]
channelMean = 0.1307
channelStd = 0.3081
lr = 0.1
numEpochs = 100
batchSize = 64

if(whichModel == "mine"):
    opt = Optimizers.GradientDescentOptimizer(lr)
    dataset = ExampleDatasets.MNISTDataset("./MNIST", 
        Transforms.NormalizeImage(channelMean, channelStd))
    model = ExampleModels.FullyConnectedClassifier(layerDims)
    model.fit(dataset, numEpochs, batchSize, opt)
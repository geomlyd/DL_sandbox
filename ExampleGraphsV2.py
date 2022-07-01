import ComputationalGraph
import CommonNodes
from importlib_metadata import Pair
import numpy as np

class FullyConnectedClassifier():

    def __init__(self, layerDimensions : list[Pair[int]], optimizer):

        self.G = ComputationalGraph.ComputationalGraph(optimizer=optimizer)

        prevLayerDimensions = None
        inputLayer = CommonNodes.InputNode()
        self.G.addNode(inputLayer, "x")        
        prevLayer = inputLayer

        for layerIndex in range(len(layerDimensions)):
            layerDims = layerDimensions[layerIndex]

            if(prevLayerDimensions is not None and 
                layerDims[0] != prevLayerDimensions[1]):
                print("Error: layer {0} has an output size of {1}, but"
                    "layer {2} has an input size of {3}".format(layerIndex - 1, 
                        prevLayerDimensions[1], layerIndex, layerDims[0]))
                exit(-1)

            W_init = np.random.normal(0.0, np.sqrt(2/layerDims[0]), 
                size=layerDims)

            linear = CommonNodes.AffineTransformation(layerDims[0], layerDims[1],
                prevLayer, W_init=W_init)
            nonlinearity = CommonNodes.ReLU(linear)
            
            self.G.addNode(linear, "linear_" + str(layerIndex))
            self.G.addNode(nonlinearity, "relu_" + str(layerIndex))
            prevLayer = nonlinearity
            prevLayerDimensions = layerDims

        y_groundTruth = CommonNodes.InputNode()
        self.G.addNode(y_groundTruth, "y_groundTruth", trainOnly=True)

        logSoftmax = CommonNodes.LogSoftmax(nonlinearity)
        self.G.addNode(logSoftmax, "logits")

        lossNode = CommonNodes.NegativeLogLikelihoodLoss(logSoftmax, y_groundTruth)
        lossOut = CommonNodes.OutputNode(lossNode)
        self.G.addNode(lossNode, "nll_loss", trainOnly=True)
        self.G.addNode(lossOut, "loss", trainOnly=True)
        
        out = CommonNodes.OutputNode(logSoftmax)
        out.trackGradients = False
        self.G.addNode(out, "output")



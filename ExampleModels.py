from typing import Tuple
from ComputationalGraph import ComputationalGraph
import Model
import numpy as np
import CommonNodes
from importlib_metadata import Pair

class FullyConnectedClassifier(Model.Model):

    def __init__(self, layerDimensions : list[Pair[int]]):

        
        self.G = ComputationalGraph()
        super().__init__(self.G)

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

    def printLossAndAccuracy(self, epoch):

        loss = self.G.getNode("loss").value
        logits = self.G.getNode("output").value
        predictions = np.argmax(logits, axis=1)
        groundTruthClasses = self.G.getNode("y_groundTruth").value
        accuracy = np.mean(groundTruthClasses == predictions)
        print("Epoch {0}: loss {1}, accuracy {2}".format(epoch, loss, accuracy))

    def loadInput(self, input: Tuple):
        self.G.getNode("x").value = np.reshape(input[0], (-1, 784))
        self.G.getNode("y_groundTruth").value = input[1]

    def __call__(self, x):
        self.G.getNode("x").value = np.reshape(x, (-1, 784))
        self.G.runForwardPass(runTraining=False)
        o = self.G.getNode("output").value
        return o#np.argmax(o, axis=1)


class FullyConnectedRegressor(Model.Model):

    def __init__(self, layerDimensions : list[Pair[int]]):

        self.G = ComputationalGraph()
        super().__init__(self.G)

        prevLayerDimensions = None
        inputLayer = CommonNodes.InputNode()
        self.G.addNode(inputLayer, "x")        
        prevLayer = inputLayer

        for layerIndex in range(len(layerDimensions)):
            layerDims = layerDimensions[layerIndex]

            if(layerIndex == len(layerDimensions) - 1 and layerDims[1] != 1):
                print("Error: final layer must have output dimension of 1")
                exit(-1)
            if(prevLayerDimensions is not None and 
                layerDims[0] != prevLayerDimensions[1]):
                print("Error: layer {0} has an output size of {1}, but"
                    "layer {2} has an input size of {3}".format(layerIndex - 1, 
                        prevLayerDimensions[1], layerIndex, layerDims[0]))
                exit(-1)

            W_init = np.random.uniform(-np.sqrt(1/layerDims[1]), np.sqrt(1/layerDims[1]), 
                size=layerDims)
            b_init = np.random.uniform(-np.sqrt(1/layerDims[1]), np.sqrt(1/layerDims[1]), layerDims[1])

            linear = CommonNodes.AffineTransformation(layerDims[0], layerDims[1],
                prevLayer, W_init=W_init, b_init=b_init)
            nonlinearity = CommonNodes.ReLU(linear)
            
            self.G.addNode(linear, "linear_" + str(layerIndex))
            self.G.addNode(nonlinearity, "relu_" + str(layerIndex))
            prevLayer = nonlinearity
            prevLayerDimensions = layerDims

        out = CommonNodes.OutputNode(nonlinearity)
        out.trackGradients = False
        self.G.addNode(out, "output")

        y_groundTruth = CommonNodes.InputNode()
        self.G.addNode(y_groundTruth, "y_groundTruth", trainOnly=True)

        diff = CommonNodes.Subtract(nonlinearity, y_groundTruth)
        sq = CommonNodes.Square(diff)
        lossNode = CommonNodes.ReduceMean(sq)
        lossOut = CommonNodes.OutputNode(lossNode)
        self.G.addNode(diff, "-", trainOnly=True)
        self.G.addNode(sq, "^2", trainOnly=True)
        self.G.addNode(lossNode, "reduce_sum", trainOnly=True)
        self.G.addNode(lossOut, "loss", trainOnly=True)

    def loadInput(self, input : Tuple):
        self.G.getNode("x").value = input[0]
        self.G.getNode("y_groundTruth").value = input[1]

    def __call__(self, x):
        self.G.getNode("x").value = x
        self.G.runForwardPass(runTraining=False)
        o = self.G.getNode("output").value
        return o

class LinearRegression(Model.Model):

    def __init__(self, inputDim):

        self.G = ComputationalGraph()
        super().__init__(self.G)

        inputLayer = CommonNodes.InputNode()
        self.G.addNode(inputLayer, "x")       
     
        lin = CommonNodes.AffineTransformation(inputDim, 1, inputLayer)
        self.G.addNode(lin, "linear") 

        out = CommonNodes.OutputNode(lin)
        out.trackGradients = False
        self.G.addNode(out, "output")

        y_groundTruth = CommonNodes.InputNode()
        self.G.addNode(y_groundTruth, "y_groundTruth", trainOnly=True)

        diff = CommonNodes.Subtract(lin, y_groundTruth)
        sq = CommonNodes.Square(diff)
        lossNode = CommonNodes.ReduceMean(sq)
        lossOut = CommonNodes.OutputNode(lossNode)
        self.G.addNode(diff, "-", trainOnly=True)
        self.G.addNode(sq, "^2", trainOnly=True)
        self.G.addNode(lossNode, "reduce_sum", trainOnly=True)
        self.G.addNode(lossOut, "loss", trainOnly=True)

    def loadInput(self, input : Tuple):
        self.G.getNode("x").value = input[0]
        self.G.getNode("y_groundTruth").value = input[1]

    def __call__(self, x):
        self.G.getNode("x").value = x
        self.G.runForwardPass(runTraining=False)
        o = self.G.getNode("output").value
        return o    
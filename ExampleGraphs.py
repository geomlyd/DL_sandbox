from cmath import inf

from importlib_metadata import Pair
import CommonNodes
import ComputationalGraph
import numpy as np

class X2logX_Graph():

    def __init__(self) -> None:
        self.G = ComputationalGraph.ComputationalGraph()

        x = CommonNodes.InputNode()
        sq = CommonNodes.Square(x)
        _log = CommonNodes.Log(x)
        mul = CommonNodes.PointwiseMul([sq, _log])
        out = CommonNodes.OutputNode(mul)
       

        self.G.addNode(x, "x")
        self.G.addNode(sq, "^2")
        self.G.addNode(_log, "log")
        self.G.addNode(mul, "*")
        self.G.addNode(out, "output")


    def __call__(self, x):
        self.G.getNode("x").value = x
        self.G.runForwardPass()
        self.G.runBackwardPass()
        o = self.G.getNode("output").value
        d = self.G.getNode("x").totalGradient
        return o, d

class InterestingGraph():

    def __init__(self) -> None:
        self.G = ComputationalGraph.ComputationalGraph()    

        x = CommonNodes.InputNode()
        c = CommonNodes.ConstantNode(1)
        inv = CommonNodes.PointwiseDivide(c, x)
        _sin = CommonNodes.Sin(inv)
        mul = CommonNodes.PointwiseMul([x, _sin])
        out = CommonNodes.OutputNode(mul)

        self.G.addNode(x, "x")
        self.G.addNode(_sin, "sin")
        self.G.addNode(c, "1")
        self.G.addNode(inv, "/")
        self.G.addNode(mul, "*")
        self.G.addNode(out, "output")        

    def __call__(self, x):
        self.G.getNode("x").value = x
        self.G.runForwardPass()
        self.G.runBackwardPass()
        o = self.G.getNode("output").value
        d = self.G.getNode("x").totalGradient
        return o, d

class LinearRegression():

    def __init__(self, inputDim, optimizer):

        self.G = ComputationalGraph.ComputationalGraph(optimizer=optimizer)

        x = CommonNodes.InputNode()

        y_groundTruth = CommonNodes.InputNode()
        y_pred = CommonNodes.AffineTransformation(inputDim, 1, x)

        diff = CommonNodes.Subtract(y_pred, y_groundTruth)
        sq = CommonNodes.Square(diff)
        lossNode = CommonNodes.ReduceSum(sq)
        lossOut = CommonNodes.OutputNode(lossNode)

        out = CommonNodes.OutputNode(y_pred)
        out.trackGradients = False

        self.G.addNode(out, "output")
        self.G.addNode(y_groundTruth, "y_groundTruth")
        self.G.addNode(y_pred, "y_pred")
        self.G.addNode(x, "x")
        self.G.addNode(diff, "-", trainOnly=True)
        self.G.addNode(sq, "^2", trainOnly=True)
        self.G.addNode(lossNode, "reduce_sum", trainOnly=True)
        self.G.addNode(lossOut, "loss", trainOnly=True)

    def __call__(self, x):
        self.G.getNode("x").value = x
        self.G.runForwardPass(runTraining=False)
        o = self.G.getNode("output").value
        return o

    def fit(self, x, targets, numEpochs):
        l = inf
        
        for i in range(numEpochs):
            self.G.getNode("x").value = x
            self.G.getNode("y_groundTruth").value = targets
            self.G.runForwardPass()
            self.G.runBackwardPass()
            l = self.G.getNode("loss").value
            print("Iteration {0} : loss {1}".format(i, l))

class FullyConnectedRegressor():

    def __init__(self, layerDimensions : list[Pair[int]], optimizer):

        self.G = ComputationalGraph.ComputationalGraph(optimizer=optimizer)

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

            W_init = np.random.normal(0.0, np.sqrt(2/layerDims[0]), 
                size=layerDims)

            linear = CommonNodes.AffineTransformation(layerDims[0], layerDims[1],
                prevLayer, W_init=W_init)
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
        lossNode = CommonNodes.ReduceSum(sq)
        lossOut = CommonNodes.OutputNode(lossNode)
        self.G.addNode(diff, "-", trainOnly=True)
        self.G.addNode(sq, "^2", trainOnly=True)
        self.G.addNode(lossNode, "reduce_sum", trainOnly=True)
        self.G.addNode(lossOut, "loss", trainOnly=True)

    def __call__(self, x):
        self.G.getNode("x").value = x
        self.G.runForwardPass(runTraining=False)
        o = self.G.getNode("output").value
        return o

    def fit(self, x, targets, numEpochs):
        l = inf
        
        for i in range(numEpochs):
            self.G.getNode("x").value = x
            self.G.getNode("y_groundTruth").value = targets
            self.G.runForwardPass()
            self.G.runBackwardPass()
            l = self.G.getNode("loss").value
            print("Iteration {0} : loss {1}".format(i, l))        


class Rotation():

    def __init__(self, degrees) -> None:
        self.G = ComputationalGraph.ComputationalGraph()

        x = CommonNodes.InputNode()
        radians = degrees*np.pi/180
        rotMatrix = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
        rotMatrix = rotMatrix.T
        rot = CommonNodes.AffineTransformation(2, 2, x,  rotMatrix, np.zeros(2))

        out = CommonNodes.OutputNode(rot)

        self.G.addNode(x, "x")
        self.G.addNode(rot, "rot")
        self.G.addNode(out, "output")


    def setDegrees(self, deg):
        radians = deg*np.pi/180
        rotMatrix = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
        rotMatrix = rotMatrix.T
        self.G.getNode("rot").W = rotMatrix

    def __call__(self, x):
        self.G.getNode("x").value = x
        self.G.runForwardPass()
        o = self.G.getNode("output").value
        return o
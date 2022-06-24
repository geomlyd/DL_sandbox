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
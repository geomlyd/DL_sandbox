import CommonNodes
import ComputationalGraph

class X2logX_Graph():

    def __init__(self) -> None:
        self.G = ComputationalGraph.ComputationalGraph()

        x = CommonNodes.InputNode()
        sq = CommonNodes.Square()
        log = CommonNodes.Log()
        out = CommonNodes.OutputNode()
        mul = CommonNodes.PointwiseMul()

        self.G.addNode(x, "x")
        self.G.addNode(sq, "^2")
        self.G.addNode(log, "log")
        self.G.addNode(mul, "*")
        self.G.addNode(out, "output")

        self.G.addEdge("x", "^2", "out", "in")
        self.G.addEdge("x", "log", "out", "in")
        self.G.addEdge("^2", "*", "in", "arg1")
        self.G.addEdge("log", "*", "in", "arg2")
        self.G.addEdge("*", "output", "in", "output")

    def __call__(self, x):
        self.G.getNode("x").setValue(x)
        self.G.runForwardPass()
        outputDict = self.G.getNode("output").getValue()
        return outputDict["output"]

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

        self.G.addEdge("x", "/")
        self.G.addEdge("1", "/")
        self.G.addEdge("/", "sin")
        self.G.addEdge("x", "*")
        self.G.addEdge("sin", "*")
        self.G.addEdge("*", "output")       

    def __call__(self, x):
        self.G.getNode("x").setValue(x)
        self.G.runForwardPass()
        o = self.G.getNode("output").getValue()
        return o     

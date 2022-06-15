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
        _sin = CommonNodes.Sin()
        inv = CommonNodes.PointwiseDivide()
        mul = CommonNodes.PointwiseMul()
        out = CommonNodes.OutputNode()
        c = CommonNodes.ConstantNode()
        c.setValue(1)

        self.G.addNode(x, "x")
        self.G.addNode(_sin, "sin")
        self.G.addNode(c, "1")
        self.G.addNode(inv, "/")
        self.G.addNode(mul, "*")
        self.G.addNode(out, "output")        

        self.G.addEdge("x", "/", "out", "denominator")
        self.G.addEdge("1", "/", "out", "numerator")
        self.G.addEdge("/", "sin", "out", "in")
        self.G.addEdge("x", "*", "out", "in1")
        self.G.addEdge("sin", "*", "out", "in2")

        self.G.addEdge("*", "output", "in", "output")       

    def __call__(self, x):
        self.G.getNode("x").setValue(x)
        self.G.runForwardPass()
        outputDict = self.G.getNode("output").getValue()
        return outputDict["output"]        
